#!/usr/bin/env python3
"""
Unified training script for GPT-2 variants.

Modes:
  - normal: Standard GPT-2 (no Q-head), LM loss only
  - qhead: GPT-2 with Q-head, using Monte Carlo returns (discounted sum of rewards)
  - gae: GPT-2 with Q-head, using GAE lambda-returns
  - avglogprob: GPT-2 with Q-head, using average log probability of future tokens
                as Q targets (intrinsic reward based on prediction confidence)

Multiple modes can be specified to run sequentially.
"""

import os
import math
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from datasets import load_dataset

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from model import GPT2WithQ
from dataset import TokenRewardDataset, collate_fn


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed: {seed}")


def estimate_batch_size(device, seq_len=128, safety_factor=0.5):
    """Estimate optimal batch size based on available GPU memory."""
    if device == "cpu" or not torch.cuda.is_available():
        return 8

    try:
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory_gb = gpu_props.total_memory / (1024**3)
        base_memory_gb = 2.0
        per_sample_gb = seq_len * 4e-6
        available_gb = (total_memory_gb - base_memory_gb) * safety_factor
        estimated_batch = int(available_gb / per_sample_gb)
        batch_size = max(1, min(16, estimated_batch))
        print(f"GPU: {gpu_props.name} ({total_memory_gb:.1f} GB), auto batch size: {batch_size}")
        return batch_size
    except Exception as e:
        print(f"Could not estimate batch size: {e}, using default=8")
        return 8


def compute_discounted_returns_batch(rewards, gamma, bootstrap_values=None):
    """Compute discounted returns with optional bootstrapping."""
    B, L = rewards.shape
    returns = torch.zeros_like(rewards)

    for i in range(B):
        r = rewards[i]
        future = bootstrap_values[i].item() if bootstrap_values is not None else 0.0
        for t in range(L - 1, -1, -1):
            returns[i, t] = future
            future = r[t] + gamma * future
    return returns


def compute_avg_logprob_returns_batch(logits, target_tokens, attention_mask):
    """Compute average log probability of future tokens as Q targets.

    For each position t, computes the average log probability of all tokens
    from position t+1 to the end of the sequence. This serves as an intrinsic
    reward signal based on the model's prediction confidence.

    Args:
        logits: (B, L, V) model logits
        target_tokens: (B, L-1) next tokens (shifted labels)
        attention_mask: (B, L-1) mask for valid positions

    Returns:
        returns: (B, L-1) average log prob of future tokens at each position
    """
    B, L_minus_1 = target_tokens.shape
    device = logits.device

    # Compute log probabilities for each position
    # shift_logits: (B, L-1, V), target_tokens: (B, L-1)
    shift_logits = logits[:, :-1, :].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, L-1, V)

    # Get log prob of actual next tokens: (B, L-1)
    token_log_probs = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)

    # Mask out padding positions
    mask = attention_mask.to(torch.float)
    token_log_probs = token_log_probs * mask

    # Compute average log prob of future tokens for each position
    # At position t, we want mean(log_probs[t:]) for all valid future positions
    returns = torch.zeros(B, L_minus_1, device=device)

    for i in range(B):
        # Find valid length for this sequence
        valid_len = int(mask[i].sum().item())
        if valid_len == 0:
            continue

        # Compute cumulative sum from the end (reverse cumsum)
        seq_log_probs = token_log_probs[i, :valid_len]
        cumsum_from_end = torch.flip(torch.cumsum(torch.flip(seq_log_probs, [0]), dim=0), [0])

        # Count of future tokens at each position (including current)
        counts = torch.arange(valid_len, 0, -1, device=device, dtype=torch.float)

        # Average log prob of tokens from position t onwards
        avg_log_probs = cumsum_from_end / counts

        returns[i, :valid_len] = avg_log_probs

    return returns


def compute_gae_returns_batch(rewards, values, gamma, gae_lambda, bootstrap_values=None):
    """Compute GAE lambda-returns."""
    B, L = rewards.shape
    device = rewards.device

    if bootstrap_values is not None:
        values_extended = torch.cat([values, bootstrap_values.unsqueeze(1)], dim=1)
    else:
        values_extended = torch.cat([values, torch.zeros(B, 1, device=device)], dim=1)

    td_errors = torch.zeros(B, L, device=device)
    for t in range(L - 1):
        td_errors[:, t] = rewards[:, t + 1] + gamma * values_extended[:, t + 2] - values_extended[:, t + 1]
    td_errors[:, L - 1] = gamma * values_extended[:, L] - values_extended[:, L]

    gae = torch.zeros(B, L, device=device)
    running_gae = torch.zeros(B, device=device)
    for t in range(L - 1, -1, -1):
        running_gae = td_errors[:, t] + gamma * gae_lambda * running_gae
        gae[:, t] = running_gae

    gae_returns = torch.zeros(B, L, device=device)
    for t in range(L - 1):
        gae_returns[:, t] = gae[:, t] + values_extended[:, t + 1]
    gae_returns[:, L - 1] = values_extended[:, L]

    return gae_returns


def train_step_qhead(model, batch, optimizer, gamma=0.99, q_loss_weight=1.0, device="cpu", gae_lambda=None, use_avg_logprob=False):
    """Training step for GPT2WithQ model (qhead, gae, or avglogprob mode)."""
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    rewards = batch["rewards"].to(device)

    logits, q_values = model(input_ids=input_ids, attention_mask=attention_mask)

    # LM loss (shifted)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    shift_labels_masked = shift_labels.clone()
    shift_labels_masked[shift_mask == 0] = -100
    lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels_masked.view(-1), ignore_index=-100)

    # Q loss
    shift_q = q_values[:, :-1, :].contiguous()
    next_tokens = shift_labels
    q_pred = shift_q.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

    # Bootstrap values
    with torch.no_grad():
        last_logits = logits[:, -1, :]
        last_q = q_values[:, -1, :]
        policy_probs = torch.softmax(last_logits, dim=-1)
        bootstrap_values = (policy_probs * last_q).sum(dim=-1)

    # Compute returns
    if use_avg_logprob:
        # Use average log probability of future tokens as Q targets
        with torch.no_grad():
            returns_target = compute_avg_logprob_returns_batch(logits, shift_labels, shift_mask)
    elif gae_lambda is not None:
        with torch.no_grad():
            all_policy_probs = torch.softmax(logits, dim=-1)
            values = (all_policy_probs * q_values).sum(dim=-1)
        returns = compute_gae_returns_batch(rewards, values, gamma, gae_lambda, bootstrap_values=bootstrap_values)
        returns_target = returns[:, :-1].contiguous()
    else:
        returns = compute_discounted_returns_batch(rewards, gamma, bootstrap_values=bootstrap_values)
        returns_target = returns[:, :-1].contiguous()

    mask = shift_mask.to(torch.float)
    q_loss_per_pos = F.mse_loss(q_pred, returns_target, reduction="none") * mask
    q_loss = q_loss_per_pos.sum() / (mask.sum() + 1e-8)

    loss = lm_loss + q_loss_weight * q_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    perplexity = math.exp(lm_loss.item()) if lm_loss.item() < 100 else float('inf')

    return {
        "loss": loss.item(),
        "lm_loss": lm_loss.item(),
        "q_loss": q_loss.item(),
        "perplexity": perplexity,
    }


def train_step_normal(model, batch, optimizer, device="cpu"):
    """Training step for standard GPT-2 (normal mode, no Q-head)."""
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    shift_labels_masked = shift_labels.clone()
    shift_labels_masked[shift_mask == 0] = -100
    lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels_masked.view(-1), ignore_index=-100)

    optimizer.zero_grad()
    lm_loss.backward()
    optimizer.step()

    perplexity = math.exp(lm_loss.item()) if lm_loss.item() < 100 else float('inf')

    return {
        "loss": lm_loss.item(),
        "lm_loss": lm_loss.item(),
        "perplexity": perplexity,
    }


class StreamingTextDataset(IterableDataset):
    """Streaming dataset that yields tokenized blocks from a HuggingFace dataset."""

    def __init__(self, dataset_name, config_name, tokenizer, block_size=128,
                 text_field="text", split="train", shuffle_buffer_size=10000, seed=None):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_field = text_field
        self.split = split
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def __iter__(self):
        ds = load_dataset(
            self.dataset_name,
            self.config_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True
        )

        if self.shuffle_buffer_size > 0:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)

        token_buffer = []

        for example in ds:
            text = example[self.text_field]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.block_size:
                block = token_buffer[:self.block_size]
                token_buffer = token_buffer[self.block_size:]
                rewards = [0.0] * len(block)
                yield {"input_ids": block, "rewards": rewards}


def build_streaming_dataset(tokenizer, dataset="wikipedia", block_size=128, seed=None):
    """Build a streaming dataset for training."""
    dataset_configs = {
        "wikipedia": ("wikimedia/wikipedia", "20231101.en", "text"),
        "openwebtext": ("Skylion007/openwebtext", None, "text"),
        "c4": ("allenai/c4", "en", "text"),
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
        seed=seed
    )


def streaming_collate_fn(batch, pad_token_id):
    """Collate function for streaming dataset batches."""
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    rewards = [torch.tensor(item["rewards"], dtype=torch.float) for item in batch]

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


def save_checkpoint(model, optimizer, step, save_dir, mode, args=None, wandb_run_id=None):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{mode}_step_{step}.pt")

    checkpoint = {
        "step": step,
        "mode": mode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if args is not None:
        checkpoint["args"] = vars(args)
    if wandb_run_id is not None:
        checkpoint["wandb_run_id"] = wandb_run_id

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """Load a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint.get("step", 0)
    args = checkpoint.get("args", None)
    wandb_run_id = checkpoint.get("wandb_run_id", None)
    print(f"Loaded checkpoint from step {step}")
    if wandb_run_id:
        print(f"W&B run ID: {wandb_run_id}")
    return step, args, wandb_run_id


def sample_with_q_tilt(model, input_ids, beta=1.0, temperature=1.0, top_k=None, top_p=None):
    """Sample next token using Q-tilted distribution."""
    model.eval()
    with torch.no_grad():
        logits, q_values = model(input_ids)
        last_logits = logits[:, -1, :]
        last_q = q_values[:, -1, :]

        tilted_logits = last_logits + last_q / beta
        tilted_logits = tilted_logits / temperature

        if top_k is not None:
            top_k = min(top_k, tilted_logits.size(-1))
            indices_to_remove = tilted_logits < torch.topk(tilted_logits, top_k)[0][..., -1, None]
            tilted_logits[indices_to_remove] = float('-inf')

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(tilted_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            tilted_logits[indices_to_remove] = float('-inf')

        probs = F.softmax(tilted_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        info = {
            "logits": last_logits,
            "q_values": last_q,
            "tilted_logits": tilted_logits,
            "probs": probs,
        }
    model.train()
    return next_token, info


def generate_with_q_tilt(model, tokenizer, prompt, max_new_tokens=50, beta=1.0,
                         temperature=1.0, top_k=None, top_p=None, device="cpu"):
    """Generate text using Q-tilted sampling."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated_tokens = []
    for _ in range(max_new_tokens):
        next_token, _ = sample_with_q_tilt(
            model, input_ids, beta=beta, temperature=temperature,
            top_k=top_k, top_p=top_p
        )
        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    model.train()
    return generated_text, generated_tokens


def find_latest_checkpoint(save_dir, mode):
    """Find the latest checkpoint for a given mode."""
    import glob
    pattern = os.path.join(save_dir, mode, f"{mode}_step_*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, 0
    # Extract step numbers and find max
    steps = []
    for ckpt in checkpoints:
        try:
            step = int(ckpt.split("_step_")[-1].replace(".pt", ""))
            steps.append((step, ckpt))
        except ValueError:
            continue
    if not steps:
        return None, 0
    steps.sort(reverse=True)
    return steps[0][1], steps[0][0]


def run_training(mode, args, tokenizer, device, resume_from=None):
    """Run training for a single mode."""
    print(f"\n{'='*60}")
    print(f"Starting training: mode={mode}")
    print(f"{'='*60}")

    # Determine save directory
    save_dir = os.path.join(args.save_dir, mode)
    os.makedirs(save_dir, exist_ok=True)

    # Create model based on mode
    use_avg_logprob = False
    if mode == "normal":
        config = GPT2Config.from_pretrained("gpt2")
        model = GPT2LMHeadModel(config)  # Random init
        model.resize_token_embeddings(len(tokenizer))
        gae_lambda = None
    elif mode == "qhead":
        config = GPT2Config.from_pretrained("gpt2")
        model = GPT2WithQ(config)  # Random init
        model.transformer.resize_token_embeddings(len(tokenizer))
        gae_lambda = None
    elif mode == "gae":
        config = GPT2Config.from_pretrained("gpt2")
        model = GPT2WithQ(config)  # Random init
        model.transformer.resize_token_embeddings(len(tokenizer))
        gae_lambda = args.gae_lambda
    elif mode == "avglogprob":
        config = GPT2Config.from_pretrained("gpt2")
        model = GPT2WithQ(config)  # Random init
        model.transformer.resize_token_embeddings(len(tokenizer))
        gae_lambda = None
        use_avg_logprob = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Handle resume
    start_step = 0
    wandb_run_id = None
    if resume_from:
        if resume_from == "auto":
            ckpt_path, ckpt_step = find_latest_checkpoint(args.save_dir, mode)
            if ckpt_path:
                print(f"Auto-resuming from: {ckpt_path}")
                start_step, _, wandb_run_id = load_checkpoint(ckpt_path, model, optimizer, device=device)
            else:
                print(f"No checkpoint found for mode={mode}, starting fresh")
        else:
            print(f"Resuming from: {resume_from}")
            start_step, _, wandb_run_id = load_checkpoint(resume_from, model, optimizer, device=device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build dataset
    dataset = build_streaming_dataset(tokenizer, dataset=args.dataset, block_size=args.max_len, seed=args.seed)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        collate_fn=lambda b: streaming_collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )

    # Initialize W&B
    use_wandb = args.wandb and HAS_WANDB
    if use_wandb:
        run_name = f"{mode}_seed{args.seed}"
        if wandb_run_id:
            # Resume existing W&B run
            print(f"Resuming W&B run: {wandb_run_id}")
            wandb.init(
                project=args.wandb_project,
                id=wandb_run_id,
                resume="must",
                reinit=True
            )
        else:
            # Start new W&B run
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "mode": mode,
                    "seed": args.seed,
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "gamma": args.gamma,
                    "gae_lambda": gae_lambda,
                    "use_avg_logprob": use_avg_logprob,
                    "max_len": args.max_len,
                },
                reinit=True
            )
        wandb_run_id = wandb.run.id  # Save for checkpoints
        print(f"W&B initialized: {wandb.run.url}")

    # Training loop
    print(f"\n{'Step':>8} | {'LM Loss':>10} {'PPL':>10}" + (" | Q Loss" if mode != "normal" else ""))
    print("-" * 50)

    global_step = start_step
    data_iter = iter(dataloader)
    pbar = tqdm(total=args.steps, initial=start_step, desc=f"Training ({mode})")

    acc = {"lm_loss": 0.0, "q_loss": 0.0, "perplexity": 0.0}
    interval_count = 0
    skipped_batches = 0
    total_batches = 0

    while global_step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Wraparound: restart iterator
            data_iter = iter(dataloader)
            batch = next(data_iter)

        total_batches += 1

        # Train step with OOM handling
        try:
            if mode == "normal":
                stats = train_step_normal(model, batch, optimizer, device=device)
            else:
                stats = train_step_qhead(model, batch, optimizer, gamma=args.gamma,
                                         q_loss_weight=args.q_weight, device=device, gae_lambda=gae_lambda,
                                         use_avg_logprob=use_avg_logprob)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                skipped_batches += 1
                skip_pct = 100.0 * skipped_batches / total_batches
                tqdm.write(f"  [OOM] Skipping batch at step {global_step} (skipped {skipped_batches}/{total_batches} = {skip_pct:.2f}%)")
                # Clear CUDA cache and skip this batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            else:
                raise  # Re-raise non-OOM errors

        acc["lm_loss"] += stats["lm_loss"]
        acc["perplexity"] += stats["perplexity"]
        if "q_loss" in stats:
            acc["q_loss"] += stats["q_loss"]

        global_step += 1
        interval_count += 1
        pbar.update(1)

        # Save checkpoint
        if global_step % args.save_interval == 0:
            save_checkpoint(model, optimizer, global_step, save_dir, mode, args=args, wandb_run_id=wandb_run_id)

        # Log
        if global_step % args.log_interval == 0 and interval_count > 0:
            avg_lm = acc["lm_loss"] / interval_count
            avg_ppl = acc["perplexity"] / interval_count
            avg_q = acc["q_loss"] / interval_count if mode != "normal" else 0

            if mode == "normal":
                tqdm.write(f"{global_step:>8} | {avg_lm:>10.4f} {avg_ppl:>10.2f}")
            else:
                tqdm.write(f"{global_step:>8} | {avg_lm:>10.4f} {avg_ppl:>10.2f} | {avg_q:.4f}")

            if use_wandb:
                skip_pct = 100.0 * skipped_batches / total_batches if total_batches > 0 else 0.0
                log_dict = {
                    "step": global_step,
                    "lm_loss": avg_lm,
                    "perplexity": avg_ppl,
                    "skipped_batch_pct": skip_pct,
                }
                if mode != "normal":
                    log_dict["q_loss"] = avg_q
                wandb.log(log_dict, step=global_step)

            acc = {"lm_loss": 0.0, "q_loss": 0.0, "perplexity": 0.0}
            interval_count = 0

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()

    # Final save
    save_checkpoint(model, optimizer, global_step, save_dir, mode, args=args, wandb_run_id=wandb_run_id)
    print(f"Training complete for mode={mode}")

    if use_wandb:
        wandb.finish()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 variants")
    parser.add_argument("--mode", type=str, nargs="+", default=["qhead"],
                        choices=["normal", "qhead", "gae", "avglogprob"],
                        help="Training mode(s). Multiple modes run sequentially.")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (auto if not specified)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--q_weight", type=float, default=1.0, help="Weight for Q loss")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda (for gae mode)")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--dataset", type=str, default="wikipedia",
                        choices=["wikipedia", "openwebtext", "c4"],
                        help="Dataset to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=10000, help="Save every N steps")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Base directory for checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="gpt2-q-head", help="W&B project name")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training. Use 'auto' to find latest checkpoint, or provide path.")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Auto batch size
    if args.batch_size is None:
        args.batch_size = estimate_batch_size(args.device, seq_len=args.max_len)

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    print(f"Modes to run: {args.mode}")
    print(f"Steps per mode: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")

    # Run each mode sequentially
    for mode in args.mode:
        run_training(mode, args, tokenizer, args.device, resume_from=args.resume)

    print("\nAll training runs complete!")


if __name__ == "__main__":
    main()
