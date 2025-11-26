#!/usr/bin/env python3
"""Analyze trained GPT2-with-Q-head checkpoints.

This script loads checkpoints and runs inference at different beta values
to analyze how Q-tilted sampling affects generation.

Usage:
    python analyze.py --checkpoint checkpoints/checkpoint_latest.pt
    python analyze.py --checkpoint checkpoints/checkpoint_step_5000.pt --prompt "The meaning of life is"
    python analyze.py --checkpoint checkpoints/checkpoint_latest.pt --compare_betas
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from model import GPT2WithQ
from train import load_checkpoint, sample_with_q_tilt, generate_with_q_tilt


def analyze_q_values(model, tokenizer, prompt, device="cpu", top_k=20):
    """Analyze Q-values for the next token given a prompt.

    Shows the top-k tokens by:
    - Original LM probability
    - Q-value
    - Q-tilted probability at various beta values
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        logits, q_values = model(input_ids)
        last_logits = logits[0, -1, :]  # (V,)
        last_q = q_values[0, -1, :]  # (V,)

        # Original probabilities
        probs = F.softmax(last_logits, dim=-1)

        # Get top-k by probability
        top_probs, top_indices = torch.topk(probs, top_k)

    print(f"\nPrompt: \"{prompt}\"")
    print(f"\n{'='*100}")
    print(f"{'Token':<20} {'P(orig)':<12} {'Q-value':<12} {'P(β=0.5)':<12} {'P(β=1.0)':<12} {'P(β=2.0)':<12}")
    print(f"{'='*100}")

    for idx in top_indices:
        token = tokenizer.decode([idx.item()])
        p_orig = probs[idx].item()
        q_val = last_q[idx].item()

        # Q-tilted probabilities at different beta values
        p_tilted = {}
        for beta in [0.5, 1.0, 2.0]:
            tilted_logits = last_logits + last_q / beta
            tilted_probs = F.softmax(tilted_logits, dim=-1)
            p_tilted[beta] = tilted_probs[idx].item()

        # Escape special characters for display
        token_display = repr(token)[1:-1]  # Remove quotes
        if len(token_display) > 18:
            token_display = token_display[:15] + "..."

        print(f"{token_display:<20} {p_orig:<12.4f} {q_val:<12.4f} {p_tilted[0.5]:<12.4f} {p_tilted[1.0]:<12.4f} {p_tilted[2.0]:<12.4f}")

    # Also show tokens with highest Q-values (may differ from highest prob)
    print(f"\n{'-'*100}")
    print("Top tokens by Q-value:")
    print(f"{'='*100}")

    _, top_q_indices = torch.topk(last_q, top_k)
    for idx in top_q_indices:
        token = tokenizer.decode([idx.item()])
        p_orig = probs[idx].item()
        q_val = last_q[idx].item()

        token_display = repr(token)[1:-1]
        if len(token_display) > 18:
            token_display = token_display[:15] + "..."

        print(f"{token_display:<20} P={p_orig:<10.4f} Q={q_val:<10.4f}")


def compare_generations_at_betas(model, tokenizer, prompt, betas, device="cpu",
                                  max_tokens=50, temperature=1.0, top_k=50, top_p=0.95):
    """Generate text at different beta values and compare.

    Args:
        model: GPT2WithQ model
        tokenizer: tokenizer
        prompt: starting prompt
        betas: list of beta values to compare
        device: device to run on
        max_tokens: max tokens to generate
        temperature: sampling temperature
        top_k: top-k filtering
        top_p: nucleus sampling threshold
    """
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Settings: temperature={temperature}, top_k={top_k}, top_p={top_p}")
    print(f"\n{'='*100}")

    for beta in betas:
        generated_text, _ = generate_with_q_tilt(
            model, tokenizer, prompt,
            max_new_tokens=max_tokens,
            beta=beta,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device
        )
        # Show only the generated part
        continuation = generated_text[len(prompt):]
        print(f"\nβ = {beta}:")
        print(f"  {continuation}")

    print(f"\n{'='*100}")


def compute_expected_q(model, tokenizer, prompt, device="cpu"):
    """Compute E_π[Q(s, a)] for different policies at current position.

    Shows how much expected future value varies with beta.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        logits, q_values = model(input_ids)
        last_logits = logits[0, -1, :]
        last_q = q_values[0, -1, :]

        print(f"\nPrompt: \"{prompt}\"")
        print(f"\n{'='*60}")
        print(f"{'Beta':<10} {'E_π[Q]':<15} {'Max Q':<15} {'Min Q':<15}")
        print(f"{'='*60}")

        betas = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
        for beta in betas:
            if beta == float('inf'):
                # Original policy
                tilted_probs = F.softmax(last_logits, dim=-1)
                beta_str = "∞ (orig)"
            else:
                tilted_logits = last_logits + last_q / beta
                tilted_probs = F.softmax(tilted_logits, dim=-1)
                beta_str = str(beta)

            expected_q = (tilted_probs * last_q).sum().item()
            max_q = last_q.max().item()
            min_q = last_q.min().item()

            print(f"{beta_str:<10} {expected_q:<15.4f} {max_q:<15.4f} {min_q:<15.4f}")

        print(f"{'='*60}")


def interactive_mode(model, tokenizer, device="cpu"):
    """Interactive mode for exploring the model."""
    print("\nInteractive mode. Commands:")
    print("  <text>         - Generate with default beta=1.0")
    print("  /beta <value>  - Set beta value")
    print("  /analyze <text>- Analyze Q-values for prompt")
    print("  /compare <text>- Compare generations at multiple betas")
    print("  /expected <text>- Show expected Q at different betas")
    print("  /quit          - Exit")
    print()

    beta = 1.0
    temperature = 0.8
    top_k = 50
    top_p = 0.95

    while True:
        try:
            user_input = input(f"[β={beta}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input[1:].split(" ", 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "quit":
                break
            elif cmd == "beta":
                try:
                    beta = float(arg)
                    print(f"Beta set to {beta}")
                except ValueError:
                    print("Invalid beta value")
            elif cmd == "temp":
                try:
                    temperature = float(arg)
                    print(f"Temperature set to {temperature}")
                except ValueError:
                    print("Invalid temperature value")
            elif cmd == "analyze":
                if arg:
                    analyze_q_values(model, tokenizer, arg, device=device)
                else:
                    print("Usage: /analyze <prompt>")
            elif cmd == "compare":
                if arg:
                    compare_generations_at_betas(
                        model, tokenizer, arg,
                        betas=[0.5, 1.0, 2.0, 5.0],
                        device=device,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                else:
                    print("Usage: /compare <prompt>")
            elif cmd == "expected":
                if arg:
                    compute_expected_q(model, tokenizer, arg, device=device)
                else:
                    print("Usage: /expected <prompt>")
            else:
                print(f"Unknown command: {cmd}")
        else:
            # Generate text
            generated, _ = generate_with_q_tilt(
                model, tokenizer, user_input,
                max_new_tokens=100,
                beta=beta,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device
            )
            print(f"\n{generated}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze GPT2-Q-head checkpoints")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt for generation")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Beta value for Q-tilted sampling")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling threshold")

    # Analysis modes
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze Q-values for the prompt")
    parser.add_argument("--compare_betas", action="store_true",
                        help="Compare generations at multiple beta values")
    parser.add_argument("--expected_q", action="store_true",
                        help="Show expected Q at different betas")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive mode")

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # Create model
    model = GPT2WithQ("gpt2")
    model.transformer.resize_token_embeddings(len(tokenizer))

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        step, saved_args = load_checkpoint(args.checkpoint, model, device=args.device)
        if saved_args:
            print(f"Checkpoint training config: {saved_args}")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("Make sure to train the model first with: python train.py")
        return

    model.to(args.device)
    model.eval()

    # Default prompts for testing
    default_prompts = [
        "The meaning of life is",
        "In the beginning,",
        "Artificial intelligence will",
        "The best way to learn is",
    ]

    if args.interactive:
        interactive_mode(model, tokenizer, device=args.device)
    elif args.prompt:
        if args.analyze:
            analyze_q_values(model, tokenizer, args.prompt, device=args.device)
        elif args.expected_q:
            compute_expected_q(model, tokenizer, args.prompt, device=args.device)
        elif args.compare_betas:
            compare_generations_at_betas(
                model, tokenizer, args.prompt,
                betas=[0.25, 0.5, 1.0, 2.0, 5.0],
                device=args.device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
        else:
            # Single generation
            generated, _ = generate_with_q_tilt(
                model, tokenizer, args.prompt,
                max_new_tokens=args.max_tokens,
                beta=args.beta,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device
            )
            print(f"\nPrompt: \"{args.prompt}\"")
            print(f"Beta: {args.beta}")
            print(f"\nGenerated:\n{generated}")
    else:
        # Run analysis on default prompts
        print("\n" + "="*80)
        print("Running analysis on default prompts")
        print("="*80)

        for prompt in default_prompts:
            print(f"\n{'='*80}")
            print(f"Prompt: \"{prompt}\"")
            print("="*80)

            compare_generations_at_betas(
                model, tokenizer, prompt,
                betas=[0.5, 1.0, 2.0, 5.0],
                device=args.device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )


if __name__ == "__main__":
    main()
