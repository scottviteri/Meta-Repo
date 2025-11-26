import math
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset

from model import GPT2WithQ
from dataset import TokenRewardDataset, collate_fn


def compute_discounted_returns_batch(rewards, gamma: float):
    """Compute discounted returns from the next-step onward.

    rewards: tensor (B, L)
    returns[t] = sum_{k=t+1}^{L-1} gamma^{k-(t+1)} * r[:, k]
    For last position returns = 0.
    """
    B, L = rewards.shape
    returns = torch.zeros_like(rewards)
    # We'll compute for each sequence by reversing and doing cumulative sums
    # For each sequence i, create r_rev = rewards[i].flip(0)
    # then compute cumulative discounted sum and flip back, then shift by 1
    device = rewards.device
    for i in range(B):
        r = rewards[i]
        # discounted future sums starting at next position
        future = 0.0
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

    # Compute return targets (discounted from next-step onward)
    returns = compute_discounted_returns_batch(rewards, gamma)
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

    return {"loss": loss.item(), "lm_loss": lm_loss.item(), "q_loss": q_loss.item()}


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


def build_wiki_dataset(tokenizer, split="train", block_size=128, fraction=None, reward_mode="zero"):
    """Load Wikipedia via Hugging Face `datasets`, tokenize, and chunk into blocks.

    Args:
        tokenizer: a GPT2 tokenizer
        split: dataset split string for `load_dataset` (e.g. "train")
        block_size: tokens per example
        fraction: if set (0-1), take that fraction of the dataset (approx.) for quicker experiments
        reward_mode: "zero" (default) or "lm_ref" (not implemented here)
    """
    # load a recent English Wikipedia snapshot
    # using the 20220301 snapshot name; if unavailable, HF will pick a matching config
    dataset_name = "wikipedia"
    config_name = "20220301.en"

    ds = load_dataset(dataset_name, config_name, split=split)
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--q_weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    dataset = build_synthetic_dataset(tokenizer, num_examples=400, max_len=48)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id))

    model = GPT2WithQ("gpt2")
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total = 0
        acc = {"loss": 0.0, "lm_loss": 0.0, "q_loss": 0.0}
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            stats = train_step(model, batch, optimizer, tokenizer, gamma=args.gamma, q_loss_weight=args.q_weight, device=args.device)
            for k, v in stats.items():
                acc[k] += v
            total += 1

        print(f"Epoch {epoch+1}: loss={acc['loss']/total:.4f}, lm={acc['lm_loss']/total:.4f}, q={acc['q_loss']/total:.4f}")


if __name__ == "__main__":
    main()
