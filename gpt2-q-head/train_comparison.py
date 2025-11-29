#!/usr/bin/env python3
"""
Comparison training script: GPT-2 with Q-head vs GPT-2 baseline (no Q-head).
Both models trained from random initialization on same data with same seed.
Logs to W&B with shared axes for easy comparison.
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from model import GPT2WithQ
from train import build_streaming_dataset, train_step, train_step_reference, streaming_collate_fn


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_random_gpt2_with_q(config):
    """Create GPT2WithQ with random initialization (not pretrained)."""
    model = GPT2WithQ(config)  # Pass config, not string, to get random init
    return model


def create_random_gpt2_baseline(config):
    """Create standard GPT2LMHeadModel with random initialization."""
    model = GPT2LMHeadModel(config)
    return model


def main():
    parser = argparse.ArgumentParser(description="Compare GPT-2 with/without Q-head from random init")
    parser.add_argument("--steps", type=int, default=50000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for Q-learning")
    parser.add_argument("--q_weight", type=float, default=1.0, help="Weight for Q loss")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=5000, help="Save checkpoints every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="gpt2-q-head-comparison")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--save_dir", type=str, default="checkpoints_comparison")
    args = parser.parse_args()

    # Set seed for reproducibility
    print(f"Setting random seed: {args.seed}")
    set_seed(args.seed)

    # Initialize W&B
    use_wandb = args.wandb and HAS_WANDB
    if use_wandb:
        run_name = args.run_name or f"comparison_seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "seed": args.seed,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "gamma": args.gamma,
                "q_weight": args.q_weight,
                "max_len": args.max_len,
                "model_type": "comparison",
            }
        )
        print(f"W&B initialized: {wandb.run.url}")

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # Create GPT-2 config (for random initialization)
    config = GPT2Config.from_pretrained("gpt2")
    config.vocab_size = len(tokenizer)  # Account for any added tokens

    # Create both models from random initialization
    print("Creating models from random initialization...")

    # Model 1: GPT-2 with Q-head
    model_q = create_random_gpt2_with_q(config)
    model_q.transformer.resize_token_embeddings(len(tokenizer))
    model_q.to(args.device)
    optimizer_q = torch.optim.AdamW(model_q.parameters(), lr=args.lr)

    # Model 2: Standard GPT-2 (no Q-head)
    model_baseline = create_random_gpt2_baseline(config)
    model_baseline.resize_token_embeddings(len(tokenizer))
    model_baseline.to(args.device)
    optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=args.lr)

    print(f"  GPT2+Q params: {sum(p.numel() for p in model_q.parameters()):,}")
    print(f"  GPT2 baseline params: {sum(p.numel() for p in model_baseline.parameters()):,}")

    # Build streaming dataset
    print("Loading streaming dataset: Wikipedia")
    dataset = build_streaming_dataset(tokenizer, dataset="wikipedia", block_size=args.max_len)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        collate_fn=lambda b: streaming_collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    print(f"\n{'='*90}")
    print(f"{'Step':>8} | {'GPT2+Q LM Loss':>14} {'GPT2+Q PPL':>12} {'Q Loss':>10} | {'Baseline LM':>12} {'Baseline PPL':>12}")
    print(f"{'='*90}")

    global_step = 0
    data_iter = iter(dataloader)
    pbar = tqdm(total=args.steps, desc="Training")

    # Accumulators for logging
    acc_q = {"lm_loss": 0.0, "q_loss": 0.0, "total_loss": 0.0}
    acc_baseline = {"lm_loss": 0.0}
    interval_count = 0

    while global_step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Train GPT-2 with Q-head
        # Note: We don't use ref_model for rewards here, just zero rewards
        stats_q = train_step(
            model_q, batch, optimizer_q, tokenizer,
            gamma=args.gamma, q_loss_weight=args.q_weight,
            device=args.device, ref_model=None, use_lm_rewards=False
        )
        acc_q["lm_loss"] += stats_q.get("lm_loss", 0)
        acc_q["q_loss"] += stats_q.get("q_loss", 0)
        acc_q["total_loss"] += stats_q.get("total_loss", 0)

        # Train baseline GPT-2 on same batch
        stats_baseline = train_step_reference(model_baseline, batch, optimizer_baseline, device=args.device)
        acc_baseline["lm_loss"] += stats_baseline.get("lm_loss", 0)

        global_step += 1
        interval_count += 1
        pbar.update(1)

        # Log every N steps
        if global_step % args.log_interval == 0 and interval_count > 0:
            avg_q_lm = acc_q["lm_loss"] / interval_count
            avg_q_loss = acc_q["q_loss"] / interval_count
            avg_baseline_lm = acc_baseline["lm_loss"] / interval_count

            ppl_q = np.exp(min(avg_q_lm, 20))  # Cap to avoid overflow
            ppl_baseline = np.exp(min(avg_baseline_lm, 20))

            # Print to console
            print(f"\r{global_step:>8} | {avg_q_lm:>14.4f} {ppl_q:>12.2f} {avg_q_loss:>10.4f} | {avg_baseline_lm:>12.4f} {ppl_baseline:>12.2f}")

            # Log to W&B with shared prefixes for comparison
            if use_wandb:
                wandb.log({
                    # GPT2+Q metrics
                    "gpt2_q/lm_loss": avg_q_lm,
                    "gpt2_q/perplexity": ppl_q,
                    "gpt2_q/q_loss": avg_q_loss,
                    "gpt2_q/total_loss": acc_q["total_loss"] / interval_count,
                    # Baseline metrics
                    "gpt2_baseline/lm_loss": avg_baseline_lm,
                    "gpt2_baseline/perplexity": ppl_baseline,
                    # Combined for easy comparison (same metric name)
                    "comparison/lm_loss_q": avg_q_lm,
                    "comparison/lm_loss_baseline": avg_baseline_lm,
                    "comparison/ppl_q": ppl_q,
                    "comparison/ppl_baseline": ppl_baseline,
                    "comparison/lm_loss_diff": avg_q_lm - avg_baseline_lm,
                    "step": global_step,
                }, step=global_step)

            # Reset accumulators
            acc_q = {"lm_loss": 0.0, "q_loss": 0.0, "total_loss": 0.0}
            acc_baseline = {"lm_loss": 0.0}
            interval_count = 0

        # Save checkpoints
        if global_step % args.save_interval == 0:
            # Save GPT2+Q
            torch.save({
                "step": global_step,
                "model_state_dict": model_q.state_dict(),
                "optimizer_state_dict": optimizer_q.state_dict(),
                "args": vars(args),
            }, os.path.join(args.save_dir, f"gpt2_q_step_{global_step}.pt"))

            # Save baseline
            torch.save({
                "step": global_step,
                "model_state_dict": model_baseline.state_dict(),
                "optimizer_state_dict": optimizer_baseline.state_dict(),
                "args": vars(args),
            }, os.path.join(args.save_dir, f"gpt2_baseline_step_{global_step}.pt"))

            print(f"  [Saved checkpoints at step {global_step}]")

    pbar.close()
    print(f"\nTraining complete! Final step: {global_step}")

    # Final save
    torch.save({
        "step": global_step,
        "model_state_dict": model_q.state_dict(),
        "optimizer_state_dict": optimizer_q.state_dict(),
        "args": vars(args),
    }, os.path.join(args.save_dir, "gpt2_q_final.pt"))

    torch.save({
        "step": global_step,
        "model_state_dict": model_baseline.state_dict(),
        "optimizer_state_dict": optimizer_baseline.state_dict(),
        "args": vars(args),
    }, os.path.join(args.save_dir, "gpt2_baseline_final.pt"))

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
