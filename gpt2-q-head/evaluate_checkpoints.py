#!/usr/bin/env python3
"""Evaluate checkpoints on held-out Wikipedia data.

This script evaluates each checkpoint in the checkpoints directory on the same
held-out Wikipedia data for fair comparison.

Usage:
    python evaluate_checkpoints.py --num_eval_samples 1000 --batch_size 16
    python evaluate_checkpoints.py --checkpoint checkpoints/checkpoint_step_10000.pt
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

from model import GPT2WithQ


def create_held_out_dataset(tokenizer, num_samples=1000, block_size=128, seed=42,
                            cache_path="eval_data/held_out_wikipedia.pt"):
    """Create or load a fixed held-out dataset for evaluation.

    Uses a different slice of Wikipedia than training (skipping first 100k documents)
    to ensure no overlap.

    Args:
        tokenizer: GPT2 tokenizer
        num_samples: Number of evaluation samples
        block_size: Tokens per sample
        seed: Random seed for reproducibility
        cache_path: Path to cache the dataset

    Returns:
        List of dicts with 'input_ids' (tensor) for each sample
    """
    # Check for cached dataset
    if os.path.exists(cache_path):
        print(f"Loading cached held-out dataset from {cache_path}")
        cached = torch.load(cache_path)
        if cached.get("num_samples") == num_samples and cached.get("block_size") == block_size:
            return cached["samples"]
        print(f"Cache mismatch (cached: {cached.get('num_samples')} samples, "
              f"requested: {num_samples}). Regenerating...")

    print(f"Creating held-out dataset with {num_samples} samples...")

    # Load Wikipedia in streaming mode with a fixed seed
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    # Skip first 100k documents to avoid overlap with training data
    # This ensures we're evaluating on data the model hasn't seen
    SKIP_DOCS = 100000
    print(f"Skipping first {SKIP_DOCS} documents to avoid training overlap...")

    samples = []
    token_buffer = []
    doc_count = 0

    for example in tqdm(ds, desc="Processing documents", total=SKIP_DOCS + num_samples * 2):
        doc_count += 1

        # Skip documents that might have been used in training
        if doc_count <= SKIP_DOCS:
            if doc_count % 10000 == 0:
                print(f"  Skipped {doc_count}/{SKIP_DOCS} documents...")
            continue

        # Tokenize the text
        text = example["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_buffer.extend(tokens)

        # Extract complete blocks
        while len(token_buffer) >= block_size and len(samples) < num_samples:
            block = token_buffer[:block_size]
            token_buffer = token_buffer[block_size:]
            samples.append({
                "input_ids": torch.tensor(block, dtype=torch.long)
            })

        if len(samples) >= num_samples:
            break

    print(f"Created {len(samples)} evaluation samples")

    # Cache the dataset
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({
        "samples": samples,
        "num_samples": num_samples,
        "block_size": block_size,
        "seed": seed,
        "skip_docs": SKIP_DOCS
    }, cache_path)
    print(f"Cached dataset to {cache_path}")

    return samples


def evaluate_checkpoint(model, eval_samples, batch_size=16, device="cuda"):
    """Evaluate a model checkpoint on held-out data.

    Args:
        model: GPT2WithQ model
        eval_samples: List of dicts with 'input_ids'
        batch_size: Batch size for evaluation
        device: Device to evaluate on

    Returns:
        Dict with evaluation metrics
    """
    model.eval()

    total_lm_loss = 0.0
    total_q_values_sum = 0.0
    total_q_values_sq_sum = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(eval_samples), batch_size), desc="Evaluating"):
            batch_samples = eval_samples[i:i+batch_size]

            # Stack input_ids
            input_ids = torch.stack([s["input_ids"] for s in batch_samples]).to(device)
            B, L = input_ids.shape

            # Forward pass
            logits, q_values = model(input_ids)

            # Compute LM loss (next-token prediction)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Per-token loss
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )

            # Q-values for taken actions
            shift_q = q_values[:, :-1, :]
            q_taken = shift_q.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

            # Top-1 accuracy
            preds = shift_logits.argmax(dim=-1)
            correct = (preds == shift_labels).sum().item()

            # Accumulate metrics
            num_tokens = (L - 1) * B
            total_lm_loss += lm_loss.item()
            total_q_values_sum += q_taken.sum().item()
            total_q_values_sq_sum += (q_taken ** 2).sum().item()
            total_tokens += num_tokens
            total_correct += correct

    # Compute aggregate metrics
    avg_lm_loss = total_lm_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_lm_loss)).item()
    avg_q = total_q_values_sum / total_tokens
    q_std = ((total_q_values_sq_sum / total_tokens) - avg_q ** 2) ** 0.5
    accuracy = total_correct / total_tokens

    return {
        "lm_loss": avg_lm_loss,
        "perplexity": perplexity,
        "q_mean": avg_q,
        "q_std": q_std,
        "top1_accuracy": accuracy,
        "num_tokens": total_tokens
    }


def get_checkpoint_step(checkpoint_path):
    """Extract step number from checkpoint filename."""
    match = re.search(r'step_(\d+)', str(checkpoint_path))
    if match:
        return int(match.group(1))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on held-out data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Evaluate a specific checkpoint (overrides checkpoint_dir)")
    parser.add_argument("--num_eval_samples", type=int, default=1000,
                        help="Number of evaluation samples")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Tokens per sample")
    parser.add_argument("--output", type=str, default="eval_results.json",
                        help="Output file for results")
    parser.add_argument("--cache_path", type=str, default="eval_data/held_out_wikipedia.pt",
                        help="Path to cache the held-out dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Create or load held-out dataset
    eval_samples = create_held_out_dataset(
        tokenizer,
        num_samples=args.num_eval_samples,
        block_size=args.block_size,
        cache_path=args.cache_path
    )

    # Find checkpoints to evaluate
    if args.checkpoint:
        checkpoints = [Path(args.checkpoint)]
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=get_checkpoint_step
        )

    print(f"Found {len(checkpoints)} checkpoints to evaluate")

    # Results storage
    results = {}

    # Evaluate each checkpoint
    for ckpt_path in checkpoints:
        step = get_checkpoint_step(ckpt_path)
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint: {ckpt_path.name} (step {step})")
        print(f"{'='*60}")

        # Load model
        model = GPT2WithQ("gpt2").to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate
        metrics = evaluate_checkpoint(
            model,
            eval_samples,
            batch_size=args.batch_size,
            device=device
        )

        # Store results
        results[step] = metrics

        # Print results
        print(f"  LM Loss: {metrics['lm_loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        print(f"  Q Mean: {metrics['q_mean']:.4f}")
        print(f"  Q Std: {metrics['q_std']:.4f}")
        print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Save results
    print(f"\n{'='*60}")
    print(f"Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Step':>8} | {'LM Loss':>8} | {'PPL':>8} | {'Q Mean':>8} | {'Q Std':>8} | {'Acc':>6}")
    print("-" * 60)

    for step in sorted(results.keys()):
        m = results[step]
        print(f"{step:>8} | {m['lm_loss']:>8.4f} | {m['perplexity']:>8.2f} | "
              f"{m['q_mean']:>8.4f} | {m['q_std']:>8.4f} | {m['top1_accuracy']:>6.4f}")


if __name__ == "__main__":
    main()
