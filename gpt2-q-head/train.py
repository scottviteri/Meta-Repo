import math
import random
import argparse
from tqdm import tqdm
from itertools import islice

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from datasets import load_dataset

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from model import GPT2WithQ
from dataset import TokenRewardDataset, collate_fn


def estimate_batch_size(device, seq_len=48, model_name="gpt2", safety_factor=0.7):
    """Estimate optimal batch size based on available GPU memory.

    Args:
        device: torch device
        seq_len: sequence length
        model_name: model identifier for size estimation
        safety_factor: fraction of memory to use (0.7 = 70%)

    Returns:
        Estimated batch size (minimum 1, maximum 64)
    """
    if device == "cpu" or not torch.cuda.is_available():
        return 8  # Conservative default for CPU

    try:
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory_gb = gpu_props.total_memory / (1024**3)

        # Rough estimates for GPT-2 small (124M params)
        # ~500MB base model, ~2KB per token per batch item for activations
        base_memory_gb = 1.0  # Model + optimizer states (with reference model)
        per_sample_gb = seq_len * 2e-6  # Rough activation memory per sample

        available_gb = (total_memory_gb - base_memory_gb) * safety_factor
        estimated_batch = int(available_gb / per_sample_gb)

        # Clamp to reasonable range
        batch_size = max(1, min(64, estimated_batch))

        print(f"GPU: {gpu_props.name} ({total_memory_gb:.1f} GB)")
        print(f"Auto batch size: {batch_size} (seq_len={seq_len})")

        return batch_size
    except Exception as e:
        print(f"Could not estimate batch size: {e}, using default=8")
        return 8


def compute_discounted_returns_batch(rewards, gamma: float, bootstrap_values=None):
    """Compute discounted returns from the next-step onward with optional bootstrapping.

    rewards: tensor (B, L)
    bootstrap_values: tensor (B,) - expected future value at the end of context window.
                      If None, assumes terminal state (bootstrap=0).

    returns[t] = sum_{k=t+1}^{L-1} gamma^{k-(t+1)} * r[:, k] + gamma^{L-1-t} * bootstrap

    This properly handles the context window boundary by bootstrapping with
    E_π[Q(s_L, a)] instead of assuming zero future value.
    """
    B, L = rewards.shape
    returns = torch.zeros_like(rewards)

    for i in range(B):
        r = rewards[i]
        # Initialize future with bootstrap value (expected value beyond context)
        future = bootstrap_values[i].item() if bootstrap_values is not None else 0.0
        for t in range(L - 1, -1, -1):
            # at position t, returns_t = discounted sum of rewards from t+1 onward
            returns[i, t] = future
            future = r[t] + gamma * future
    return returns


def train_step(model, batch, optimizer, tokenizer, gamma=0.99, q_loss_weight=1.0, device="cpu"):
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    rewards = batch["rewards"].to(device)

    batch_size, seq_len = input_ids.shape

    logits, q_values = model(input_ids=input_ids, attention_mask=attention_mask)

    # Prepare targets for next-token LM (shifted)
    lm_labels = input_ids.clone()
    # For causal LM training we set label of position t to the token at t (GPT2 expects next-token prediction with shifting)
    # We'll compute loss against logits at positions 0..L-2 predicting tokens 1..L-1.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = lm_labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    # replace padding positions in shift_labels with -100 so they are ignored
    shift_labels_masked = shift_labels.clone()
    shift_labels_masked[shift_mask == 0] = -100

    lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels_masked.view(-1))

    # Q loss: supervise the Q(s_t, a_t) for the observed action a_t (which is the next token)
    # Predicted q for positions 0..L-2
    shift_q = q_values[:, :-1, :].contiguous()
    next_tokens = shift_labels  # (B, L-1)

    # Gather predicted Q for the taken actions
    q_pred = shift_q.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    # Compute bootstrap values at the context boundary: E_π[Q(s_L, a)]
    # This is the expected Q-value under the policy distribution at the last position,
    # which provides a proper value estimate instead of assuming zero future value.
    with torch.no_grad():
        # Policy distribution at last position: π(a|s_{L-1}) = softmax(logits[:, -1, :])
        last_logits = logits[:, -1, :]  # (B, V)
        last_q = q_values[:, -1, :]  # (B, V)
        policy_probs = torch.softmax(last_logits, dim=-1)  # (B, V)
        # Expected Q under policy: V = Σ_a π(a|s) * Q(s, a)
        bootstrap_values = (policy_probs * last_q).sum(dim=-1)  # (B,)

    # Compute return targets with bootstrapping at context boundary
    returns = compute_discounted_returns_batch(rewards, gamma, bootstrap_values=bootstrap_values)
    returns_target = returns[:, :-1].contiguous()  # align with positions where next token exists

    # Mask out padded positions
    mask = shift_mask.to(torch.float)
    # MSE over masked positions
    mse = nn.MSELoss(reduction="none")
    q_loss_per_pos = mse(q_pred, returns_target) * mask
    # average only over valid positions
    denom = mask.sum()
    q_loss = q_loss_per_pos.sum() / (denom + 1e-8)

    loss = lm_loss + q_loss_weight * q_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Perplexity = exp(cross-entropy loss)
    perplexity = math.exp(lm_loss.item()) if lm_loss.item() < 100 else float('inf')

    return {
        "loss": loss.item(),
        "lm_loss": lm_loss.item(),
        "q_loss": q_loss.item(),
        "perplexity": perplexity,
    }


def train_step_reference(model, batch, optimizer, device="cpu"):
    """Training step for reference GPT-2 model (no Q-head, LM only)."""
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Compute LM loss (shifted)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    shift_labels_masked = shift_labels.clone()
    shift_labels_masked[shift_mask == 0] = -100

    lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels_masked.view(-1))

    optimizer.zero_grad()
    lm_loss.backward()
    optimizer.step()

    perplexity = math.exp(lm_loss.item()) if lm_loss.item() < 100 else float('inf')

    return {
        "lm_loss": lm_loss.item(),
        "perplexity": perplexity,
    }


@torch.no_grad()
def compute_position_diagnostics(model, dataloader, tokenizer, gamma=0.99, device="cpu", max_batches=50):
    """Compute Q prediction error and expected return as a function of position.

    Returns:
        dict with keys:
            - positions: list of position indices
            - mean_q_error: mean |Q_pred - G_target| at each position
            - mean_return: mean G_target (expected future return) at each position
            - std_q_error: std of Q error at each position
            - std_return: std of return at each position
    """
    model.eval()

    # Collect per-position statistics
    # We'll use a dict of lists, then convert to arrays
    max_len = 256  # Maximum sequence length to track
    q_errors_by_pos = [[] for _ in range(max_len)]
    returns_by_pos = [[] for _ in range(max_len)]
    q_preds_by_pos = [[] for _ in range(max_len)]

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        rewards = batch["rewards"].to(device)

        batch_size, seq_len = input_ids.shape

        logits, q_values = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute bootstrap values
        last_logits = logits[:, -1, :]
        last_q = q_values[:, -1, :]
        policy_probs = torch.softmax(last_logits, dim=-1)
        bootstrap_values = (policy_probs * last_q).sum(dim=-1)

        # Compute returns
        returns = compute_discounted_returns_batch(rewards, gamma, bootstrap_values=bootstrap_values)

        # Get Q predictions for taken actions
        shift_q = q_values[:, :-1, :]
        next_tokens = input_ids[:, 1:]
        q_pred = shift_q.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

        # Returns target aligned with Q predictions
        returns_target = returns[:, :-1]
        mask = attention_mask[:, 1:]

        # Collect statistics per position
        for b in range(batch_size):
            for t in range(seq_len - 1):
                if mask[b, t] > 0 and t < max_len:
                    q_err = abs(q_pred[b, t].item() - returns_target[b, t].item())
                    q_errors_by_pos[t].append(q_err)
                    returns_by_pos[t].append(returns_target[b, t].item())
                    q_preds_by_pos[t].append(q_pred[b, t].item())

    # Compute statistics for positions with data
    positions = []
    mean_q_error = []
    std_q_error = []
    mean_return = []
    std_return = []
    mean_q_pred = []

    for pos in range(max_len):
        if len(q_errors_by_pos[pos]) > 0:
            positions.append(pos)
            errors = q_errors_by_pos[pos]
            rets = returns_by_pos[pos]
            preds = q_preds_by_pos[pos]

            mean_q_error.append(sum(errors) / len(errors))
            std_q_error.append((sum((e - mean_q_error[-1])**2 for e in errors) / len(errors))**0.5 if len(errors) > 1 else 0)
            mean_return.append(sum(rets) / len(rets))
            std_return.append((sum((r - mean_return[-1])**2 for r in rets) / len(rets))**0.5 if len(rets) > 1 else 0)
            mean_q_pred.append(sum(preds) / len(preds))

    model.train()

    return {
        "positions": positions,
        "mean_q_error": mean_q_error,
        "std_q_error": std_q_error,
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_q_pred": mean_q_pred,
    }


def plot_position_diagnostics(diag, save_path="q_diagnostics.png"):
    """Plot Q error and expected return as a function of position."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    positions = diag["positions"]

    # Plot 1: Expected return (G_t) and Q prediction vs position
    ax1 = axes[0]
    ax1.plot(positions, diag["mean_return"], 'b-', label='Target Return (G_t)', linewidth=2)
    ax1.fill_between(positions,
                     [m - s for m, s in zip(diag["mean_return"], diag["std_return"])],
                     [m + s for m, s in zip(diag["mean_return"], diag["std_return"])],
                     alpha=0.3, color='blue')
    ax1.plot(positions, diag["mean_q_pred"], 'r--', label='Q Prediction', linewidth=2)
    ax1.set_ylabel('Value')
    ax1.set_title('Expected Return and Q Prediction vs Context Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Q prediction error vs position
    ax2 = axes[1]
    ax2.plot(positions, diag["mean_q_error"], 'g-', label='Mean |Q - G|', linewidth=2)
    ax2.fill_between(positions,
                     [max(0, m - s) for m, s in zip(diag["mean_q_error"], diag["std_q_error"])],
                     [m + s for m, s in zip(diag["mean_q_error"], diag["std_q_error"])],
                     alpha=0.3, color='green')
    ax2.set_xlabel('Context Position')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Q Prediction Error vs Context Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved position diagnostics plot to: {save_path}")


def build_synthetic_dataset(tokenizer, num_examples=200, max_len=32):
    # Generate random short sequences from some example sentences and random rewards
    texts = [
        "The cat sat on the mat.",
        "Deep learning models predict tokens.",
        "Reinforcement signals guide behavior.",
        "This is a synthetic example sequence.",
    ]

    examples = []
    for _ in range(num_examples):
        t = random.choice(texts)
        enc = tokenizer.encode(t)
        if len(enc) > max_len:
            enc = enc[:max_len]
        # make a small variation by appending a short suffix from another text
        if random.random() < 0.5:
            s = random.choice(texts)
            enc += tokenizer.encode(s)[: random.randint(0, 4)]
        L = len(enc)
        # Random small rewards (float) for each token; make last token occasionally high reward
        rewards = [random.random() * 0.1 for _ in range(L)]
        if random.random() < 0.05:
            rewards[-1] += 1.0
        examples.append({"input_ids": enc, "rewards": rewards})

    return TokenRewardDataset(examples)


class StreamingTextDataset(IterableDataset):
    """Streaming dataset that yields tokenized blocks from a HuggingFace dataset.

    Streams data continuously without loading entire dataset into memory.
    Each item is a dict with 'input_ids' and 'rewards'.
    """

    def __init__(self, dataset_name, config_name, tokenizer, block_size=128,
                 text_field="text", reward_mode="zero", split="train"):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_field = text_field
        self.reward_mode = reward_mode
        self.split = split

    def __iter__(self):
        # Load dataset in streaming mode
        ds = load_dataset(
            self.dataset_name,
            self.config_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True
        )

        token_buffer = []

        for example in ds:
            # Tokenize the text
            text = example[self.text_field]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)

            # Yield complete blocks
            while len(token_buffer) >= self.block_size:
                block = token_buffer[:self.block_size]
                token_buffer = token_buffer[self.block_size:]

                # Generate rewards (zero by default)
                if self.reward_mode == "zero":
                    rewards = [0.0] * len(block)
                else:
                    rewards = [0.0] * len(block)

                yield {"input_ids": block, "rewards": rewards}


def build_streaming_dataset(tokenizer, dataset="wikipedia", block_size=128, split="train"):
    """Build a streaming dataset for continuous training.

    Args:
        tokenizer: GPT2 tokenizer
        dataset: Dataset name. Options:
            - "wikipedia": wikimedia/wikipedia (English)
            - "pile": EleutherAI/pile (if accessible)
            - "openwebtext": Skylion007/openwebtext
            - "c4": allenai/c4
        block_size: tokens per example
        split: dataset split

    Returns:
        StreamingTextDataset instance
    """
    dataset_configs = {
        "wikipedia": ("wikimedia/wikipedia", "20231101.en", "text"),
        "openwebtext": ("Skylion007/openwebtext", None, "text"),
        "c4": ("allenai/c4", "en", "text"),
        "pile": ("EleutherAI/pile", None, "text"),
    }

    if dataset not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset}. Options: {list(dataset_configs.keys())}")

    ds_name, config, text_field = dataset_configs[dataset]

    print(f"Loading streaming dataset: {ds_name}" + (f" ({config})" if config else ""))

    return StreamingTextDataset(
        dataset_name=ds_name,
        config_name=config,
        tokenizer=tokenizer,
        block_size=block_size,
        text_field=text_field,
        split=split
    )


def streaming_collate_fn(batch, pad_token_id):
    """Collate function for streaming dataset batches."""
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    rewards = [torch.tensor(item["rewards"], dtype=torch.float) for item in batch]

    # Pad sequences (though streaming should yield fixed-size blocks)
    max_len = max(len(ids) for ids in input_ids)
    padded_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_rewards = torch.zeros((len(batch), max_len), dtype=torch.float)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, (ids, rew) in enumerate(zip(input_ids, rewards)):
        padded_ids[i, :len(ids)] = ids
        padded_rewards[i, :len(rew)] = rew
        attention_mask[i, :len(ids)] = 1

    return {
        "input_ids": padded_ids,
        "rewards": padded_rewards,
        "attention_mask": attention_mask,
    }


def build_wiki_dataset(tokenizer, split="train", block_size=128, fraction=None, reward_mode="zero"):
    """Load Wikipedia via Hugging Face `datasets`, tokenize, and chunk into blocks.

    NOTE: For large-scale training, prefer build_streaming_dataset() instead.

    Args:
        tokenizer: a GPT2 tokenizer
        split: dataset split string for `load_dataset` (e.g. "train")
        block_size: tokens per example
        fraction: if set (0-1), take that fraction of the dataset (approx.) for quicker experiments
        reward_mode: "zero" (default) or "lm_ref" (not implemented here)
    """
    # load a recent English Wikipedia snapshot
    dataset_name = "wikimedia/wikipedia"
    config_name = "20231101.en"

    ds = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
    if fraction is not None and 0.0 < fraction < 1.0:
        # take an approximate fraction by slicing
        total = len(ds)
        take = max(1, int(total * fraction))
        ds = ds.select(range(take))

    # Tokenize the raw text field (which is `text` for wikipedia)
    def tokenize_batch(batch):
        out = tokenizer(batch["text"], return_special_tokens_mask=False)
        return out

    tokenized = ds.map(tokenize_batch, batched=True, remove_columns=ds.column_names)

    # Group into blocks of block_size tokens
    def group_texts(examples):
        # concatenate all input_ids
        concatenated = sum(examples["input_ids"], [])
        total_length = (len(concatenated) // block_size) * block_size
        result = {}
        input_ids = [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
        result["input_ids"] = input_ids
        return result

    # Use a modest batch size for grouping to avoid excessive memory use
    grouped = tokenized.map(group_texts, batched=True, batch_size=1000, remove_columns=tokenized.column_names)

    # Build examples with zero rewards (by default)
    examples = []
    for row in grouped:
        ids = row["input_ids"]
        rewards = [0.0] * len(ids)
        examples.append({"input_ids": ids, "rewards": rewards})

    return TokenRewardDataset(examples)


def main():
    parser = argparse.ArgumentParser()
    # Training mode: steps (streaming) or epochs (finite dataset)
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps (for streaming mode)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (for finite dataset mode)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (auto-scaled if not specified)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--q_weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_reference", action="store_true", help="Skip training reference model")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length (block size)")
    parser.add_argument("--plot_diagnostics", action="store_true", help="Plot Q diagnostics at end of training")
    parser.add_argument("--plot_path", type=str, default="q_diagnostics.png", help="Path for diagnostics plot")
    # Dataset options
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "wikipedia", "openwebtext", "c4", "pile"],
                        help="Dataset to use for training")
    args = parser.parse_args()

    # Default to steps mode for streaming datasets, epochs for synthetic
    if args.steps is None and args.epochs is None:
        if args.dataset == "synthetic":
            args.epochs = 3
        else:
            args.steps = 10000  # Default 10k steps for streaming

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # Auto-scale batch size based on GPU memory if not specified
    if args.batch_size is None:
        args.batch_size = estimate_batch_size(args.device, seq_len=args.max_len)
    else:
        print(f"Using specified batch size: {args.batch_size}")

    # Build dataset based on selection
    if args.dataset == "synthetic":
        print("Using synthetic dataset (epoch-based training)")
        dataset = build_synthetic_dataset(tokenizer, num_examples=400, max_len=args.max_len)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id)
        )
        streaming = False
    else:
        print(f"Using {args.dataset} dataset (streaming, step-based training)")
        dataset = build_streaming_dataset(tokenizer, dataset=args.dataset, block_size=args.max_len)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size,
            collate_fn=lambda b: streaming_collate_fn(b, pad_token_id=tokenizer.pad_token_id)
        )
        streaming = True

    # Model with Q-head
    model = GPT2WithQ("gpt2")
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Reference model (standard GPT-2, no Q-head) for comparison
    if not args.no_reference:
        ref_model = GPT2LMHeadModel.from_pretrained("gpt2")
        ref_model.resize_token_embeddings(len(tokenizer))
        ref_model.to(args.device)
        ref_optimizer = torch.optim.AdamW(ref_model.parameters(), lr=args.lr)
        print("Training Q-head model and reference GPT-2 in lockstep...")
    else:
        ref_model = None
        ref_optimizer = None
        print("Training Q-head model only...")

    print(f"{'='*80}")
    print(f"{'Step':>8} | {'Q-Model PPL':>12} {'Q-Model LM':>12} {'Q-Loss':>10} | {'Ref PPL':>12} {'Ref LM':>10}")
    print(f"{'='*80}")

    # Unified training loop for both streaming and epoch-based modes
    global_step = 0
    acc_q = {"loss": 0.0, "lm_loss": 0.0, "q_loss": 0.0, "perplexity": 0.0}
    acc_ref = {"lm_loss": 0.0, "perplexity": 0.0}
    interval_count = 0

    if streaming or args.steps is not None:
        # Step-based training (streaming mode)
        total_steps = args.steps or 10000
        data_iter = iter(dataloader)
        pbar = tqdm(total=total_steps, desc="Training")

        while global_step < total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart iterator if dataset exhausted (shouldn't happen with streaming)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Train Q-head model
            stats_q = train_step(model, batch, optimizer, tokenizer, gamma=args.gamma, q_loss_weight=args.q_weight, device=args.device)
            for k, v in stats_q.items():
                acc_q[k] += v

            # Train reference model on same batch
            if ref_model is not None:
                stats_ref = train_step_reference(ref_model, batch, ref_optimizer, device=args.device)
                for k, v in stats_ref.items():
                    acc_ref[k] += v

            global_step += 1
            interval_count += 1
            pbar.update(1)

            # Log diagnostics every N steps
            if global_step % args.log_interval == 0:
                q_ppl = acc_q["perplexity"] / interval_count
                q_lm = acc_q["lm_loss"] / interval_count
                q_loss = acc_q["q_loss"] / interval_count
                if ref_model is not None:
                    ref_ppl = acc_ref["perplexity"] / interval_count
                    ref_lm = acc_ref["lm_loss"] / interval_count
                    ppl_diff = q_ppl - ref_ppl
                    tqdm.write(f"{global_step:>8} | {q_ppl:>12.2f} {q_lm:>12.4f} {q_loss:>10.4f} | {ref_ppl:>12.2f} {ref_lm:>10.4f}  (Δppl={ppl_diff:+.2f})")
                else:
                    tqdm.write(f"{global_step:>8} | {q_ppl:>12.2f} {q_lm:>12.4f} {q_loss:>10.4f} |")
                # Reset interval accumulators
                acc_q = {"loss": 0.0, "lm_loss": 0.0, "q_loss": 0.0, "perplexity": 0.0}
                acc_ref = {"lm_loss": 0.0, "perplexity": 0.0}
                interval_count = 0

        pbar.close()
    else:
        # Epoch-based training (finite dataset)
        for epoch in range(args.epochs):
            epoch_q = {"loss": 0.0, "lm_loss": 0.0, "q_loss": 0.0, "perplexity": 0.0}
            epoch_ref = {"lm_loss": 0.0, "perplexity": 0.0}
            epoch_count = 0

            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
                # Train Q-head model
                stats_q = train_step(model, batch, optimizer, tokenizer, gamma=args.gamma, q_loss_weight=args.q_weight, device=args.device)
                for k, v in stats_q.items():
                    epoch_q[k] += v
                    acc_q[k] += v

                # Train reference model on same batch
                if ref_model is not None:
                    stats_ref = train_step_reference(ref_model, batch, ref_optimizer, device=args.device)
                    for k, v in stats_ref.items():
                        epoch_ref[k] += v
                        acc_ref[k] += v

                global_step += 1
                epoch_count += 1
                interval_count += 1

                # Log diagnostics every N batches
                if global_step % args.log_interval == 0:
                    q_ppl = acc_q["perplexity"] / interval_count
                    q_lm = acc_q["lm_loss"] / interval_count
                    q_loss = acc_q["q_loss"] / interval_count
                    if ref_model is not None:
                        ref_ppl = acc_ref["perplexity"] / interval_count
                        ref_lm = acc_ref["lm_loss"] / interval_count
                        ppl_diff = q_ppl - ref_ppl
                        tqdm.write(f"{global_step:>8} | {q_ppl:>12.2f} {q_lm:>12.4f} {q_loss:>10.4f} | {ref_ppl:>12.2f} {ref_lm:>10.4f}  (Δppl={ppl_diff:+.2f})")
                    else:
                        tqdm.write(f"{global_step:>8} | {q_ppl:>12.2f} {q_lm:>12.4f} {q_loss:>10.4f} |")
                    acc_q = {"loss": 0.0, "lm_loss": 0.0, "q_loss": 0.0, "perplexity": 0.0}
                    acc_ref = {"lm_loss": 0.0, "perplexity": 0.0}
                    interval_count = 0

            # Epoch summary
            print(f"{'-'*80}")
            q_ppl = epoch_q["perplexity"] / epoch_count
            q_lm = epoch_q["lm_loss"] / epoch_count
            q_loss = epoch_q["q_loss"] / epoch_count
            if ref_model is not None:
                ref_ppl = epoch_ref["perplexity"] / epoch_count
                ref_lm = epoch_ref["lm_loss"] / epoch_count
                ppl_diff = q_ppl - ref_ppl
                print(f"Epoch {epoch+1} Summary:")
                print(f"  Q-Model:   PPL={q_ppl:.2f}, LM Loss={q_lm:.4f}, Q Loss={q_loss:.4f}")
                print(f"  Reference: PPL={ref_ppl:.2f}, LM Loss={ref_lm:.4f}")
                print(f"  Δ PPL (Q - Ref): {ppl_diff:+.2f}")
            else:
                print(f"Epoch {epoch+1}: PPL={q_ppl:.2f}, LM Loss={q_lm:.4f}, Q Loss={q_loss:.4f}")
            print(f"{'='*80}")

    # Generate position-wise diagnostics at end of training
    if args.plot_diagnostics or True:  # Always compute, optionally plot
        print("\nComputing position-wise Q diagnostics...")
        diag = compute_position_diagnostics(
            model, dataloader, tokenizer,
            gamma=args.gamma, device=args.device, max_batches=50
        )

        # Print summary statistics
        print(f"\nPosition-wise diagnostics summary:")
        print(f"  Positions analyzed: {len(diag['positions'])}")
        if len(diag['positions']) > 0:
            print(f"  Return at pos 0:  {diag['mean_return'][0]:.4f} ± {diag['std_return'][0]:.4f}")
            mid = len(diag['positions']) // 2
            print(f"  Return at pos {diag['positions'][mid]}:  {diag['mean_return'][mid]:.4f} ± {diag['std_return'][mid]:.4f}")
            print(f"  Return at pos {diag['positions'][-1]}: {diag['mean_return'][-1]:.4f} ± {diag['std_return'][-1]:.4f}")
            print(f"  Mean Q error across positions: {sum(diag['mean_q_error'])/len(diag['mean_q_error']):.4f}")

        if args.plot_diagnostics:
            plot_position_diagnostics(diag, save_path=args.plot_path)


if __name__ == "__main__":
    main()
