import os
import math
import random
import argparse
from tqdm import tqdm
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from datasets import load_dataset

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from model import GPT2WithQ
from dataset import TokenRewardDataset, collate_fn


# =============================================================================
# Q-Tilted Sampling for Inference
# =============================================================================

def sample_with_q_tilt(model, input_ids, beta=1.0, temperature=1.0, top_k=None, top_p=None):
    """Sample next token using Q-tilted distribution.

    The tilted distribution is derived from KL-constrained optimization:
        max_π' E_a~π'[Q(s,a)] - β·KL(π'||π)

    This has closed-form solution:
        π'(a|s) ∝ π(a|s) · exp(Q(s,a)/β)

    In logit space:
        logits' = logits + Q/β

    Args:
        model: GPT2WithQ model
        input_ids: tensor (B, L) of input token ids
        beta: KL penalty weight. Higher β = closer to original policy.
              Lower β = more Q-seeking. β→∞ recovers π, β→0 picks argmax Q.
        temperature: softmax temperature for final sampling
        top_k: if set, restrict to top-k tokens before sampling
        top_p: if set, use nucleus sampling with this threshold

    Returns:
        next_token: tensor (B,) of sampled token ids
        info: dict with logits, q_values, tilted_logits for analysis
    """
    model.eval()
    with torch.no_grad():
        logits, q_values = model(input_ids)

        # Get logits and Q-values for the last position
        last_logits = logits[:, -1, :]  # (B, V)
        last_q = q_values[:, -1, :]  # (B, V)

        # Q-tilted logits: logits' = logits + Q/β
        # When β is large, we stay close to the LM policy
        # When β is small, we weight heavily toward high-Q actions
        tilted_logits = last_logits + last_q / beta

        # Apply temperature
        tilted_logits = tilted_logits / temperature

        # Optional: top-k filtering
        if top_k is not None:
            top_k = min(top_k, tilted_logits.size(-1))
            indices_to_remove = tilted_logits < torch.topk(tilted_logits, top_k)[0][..., -1, None]
            tilted_logits[indices_to_remove] = float('-inf')

        # Optional: nucleus (top-p) sampling
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(tilted_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            tilted_logits[indices_to_remove] = float('-inf')

        # Sample from tilted distribution
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
    """Generate text using Q-tilted sampling.

    Args:
        model: GPT2WithQ model
        tokenizer: tokenizer
        prompt: string prompt to continue
        max_new_tokens: number of tokens to generate
        beta: KL penalty (higher = more like base LM, lower = more Q-seeking)
        temperature: sampling temperature
        top_k: top-k filtering
        top_p: nucleus sampling threshold
        device: device to run on

    Returns:
        generated_text: the full generated string
        tokens: list of generated token ids
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated_tokens = []
    for _ in range(max_new_tokens):
        next_token, _ = sample_with_q_tilt(
            model, input_ids, beta=beta, temperature=temperature,
            top_k=top_k, top_p=top_p
        )
        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    model.train()
    return generated_text, generated_tokens


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


def train_step(model, batch, optimizer, tokenizer, gamma=0.99, q_loss_weight=1.0, device="cpu",
                ref_model=None, use_lm_rewards=False):
    """Training step for GPT2WithQ model.

    Args:
        model: GPT2WithQ model to train
        batch: dict with input_ids, attention_mask, rewards
        optimizer: optimizer for model
        tokenizer: tokenizer (unused but kept for API compatibility)
        gamma: discount factor
        q_loss_weight: weight for Q loss term
        device: device to run on
        ref_model: if provided and use_lm_rewards=True, compute rewards as log P(next_token)
        use_lm_rewards: if True, use log-probs from ref_model as rewards instead of batch rewards
    """
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Compute rewards: either from batch or from reference model log-probs
    if use_lm_rewards and ref_model is not None:
        with torch.no_grad():
            ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits
            # Log-probs at each position for predicting the next token
            log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)  # (B, L-1, V)
            # Gather log prob of the actual next token
            next_tokens = input_ids[:, 1:]  # (B, L-1)
            rewards_lm = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
            # Pad to full length (last position has no next token, set reward to 0)
            rewards = torch.zeros_like(input_ids, dtype=torch.float)
            rewards[:, :-1] = rewards_lm
    else:
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

    # Compute comprehensive metrics for monitoring
    with torch.no_grad():
        pos_mask_sum = mask.sum(dim=0)  # (L-1,) - count of valid samples at each position

        # === Position-wise Q loss ===
        q_loss_by_pos = (q_loss_per_pos.sum(dim=0) / (pos_mask_sum + 1e-8)).cpu().numpy()  # (L-1,)

        # === Position-wise returns variance ===
        masked_returns = returns_target * mask
        returns_mean_by_pos = masked_returns.sum(dim=0) / (pos_mask_sum + 1e-8)  # (L-1,)
        returns_sq_diff = ((returns_target - returns_mean_by_pos.unsqueeze(0)) ** 2) * mask
        returns_var_by_pos = (returns_sq_diff.sum(dim=0) / (pos_mask_sum + 1e-8)).cpu().numpy()  # (L-1,)

        # === Position-wise mean Q predictions ===
        masked_q_pred = q_pred * mask
        q_pred_mean_by_pos = (masked_q_pred.sum(dim=0) / (pos_mask_sum + 1e-8)).cpu().numpy()  # (L-1,)

        # === Position-wise Q prediction variance ===
        q_pred_mean_expanded = (masked_q_pred.sum(dim=0) / (pos_mask_sum + 1e-8)).unsqueeze(0)
        q_pred_sq_diff = ((q_pred - q_pred_mean_expanded) ** 2) * mask
        q_pred_var_by_pos = (q_pred_sq_diff.sum(dim=0) / (pos_mask_sum + 1e-8)).cpu().numpy()  # (L-1,)

        # === Position-wise mean target returns ===
        returns_mean_by_pos_np = returns_mean_by_pos.cpu().numpy()  # (L-1,)

        # === Position-wise LM loss (cross-entropy at each position) ===
        lm_loss_per_pos = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels_masked.view(-1),
            reduction='none',
            ignore_index=-100
        ).view(batch_size, -1)  # (B, L-1)
        lm_loss_by_pos = (lm_loss_per_pos.sum(dim=0) / (pos_mask_sum + 1e-8)).cpu().numpy()  # (L-1,)

        # === Position-wise LM entropy (uncertainty in predictions) ===
        lm_probs = F.softmax(shift_logits, dim=-1)  # (B, L-1, V)
        lm_log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, L-1, V)
        lm_entropy = -(lm_probs * lm_log_probs).sum(dim=-1)  # (B, L-1)
        lm_entropy_by_pos = ((lm_entropy * mask).sum(dim=0) / (pos_mask_sum + 1e-8)).cpu().numpy()  # (L-1,)

        # === Position-wise top-1 accuracy ===
        top1_correct = (shift_logits.argmax(dim=-1) == shift_labels).float() * mask  # (B, L-1)
        top1_acc_by_pos = (top1_correct.sum(dim=0) / (pos_mask_sum + 1e-8)).cpu().numpy()  # (L-1,)

        # === Position-wise Q entropy (uncertainty in Q-tilted policy) ===
        # Q-tilted logits: logits + Q (with beta=1)
        tilted_logits = shift_logits + shift_q  # (B, L-1, V)
        tilted_probs = F.softmax(tilted_logits, dim=-1)
        tilted_log_probs = F.log_softmax(tilted_logits, dim=-1)
        tilted_entropy = -(tilted_probs * tilted_log_probs).sum(dim=-1)  # (B, L-1)
        tilted_entropy_by_pos = ((tilted_entropy * mask).sum(dim=0) / (pos_mask_sum + 1e-8)).cpu().numpy()

        # === Global Q statistics ===
        valid_q_pred = q_pred[mask > 0]
        q_pred_global_mean = valid_q_pred.mean().item() if valid_q_pred.numel() > 0 else 0.0
        q_pred_global_std = valid_q_pred.std().item() if valid_q_pred.numel() > 1 else 0.0
        q_pred_global_min = valid_q_pred.min().item() if valid_q_pred.numel() > 0 else 0.0
        q_pred_global_max = valid_q_pred.max().item() if valid_q_pred.numel() > 0 else 0.0

        # === Global returns statistics ===
        valid_returns = returns_target[mask > 0]
        returns_global_mean = valid_returns.mean().item() if valid_returns.numel() > 0 else 0.0
        returns_global_std = valid_returns.std().item() if valid_returns.numel() > 1 else 0.0
        returns_global_min = valid_returns.min().item() if valid_returns.numel() > 0 else 0.0
        returns_global_max = valid_returns.max().item() if valid_returns.numel() > 0 else 0.0

        # === Bootstrap value statistics ===
        bootstrap_mean = bootstrap_values.mean().item()
        bootstrap_std = bootstrap_values.std().item() if bootstrap_values.numel() > 1 else 0.0

        # === Reward statistics ===
        valid_rewards = rewards[attention_mask > 0]
        rewards_mean = valid_rewards.mean().item() if valid_rewards.numel() > 0 else 0.0
        rewards_std = valid_rewards.std().item() if valid_rewards.numel() > 1 else 0.0
        rewards_nonzero_frac = (valid_rewards.abs() > 1e-8).float().mean().item() if valid_rewards.numel() > 0 else 0.0

        # === Q-value distribution for taken vs not-taken actions ===
        # Q for taken action is q_pred, Q for all actions is shift_q
        # Compute mean Q over all actions efficiently without expanding mask
        # Mean over vocab dimension first, then masked mean over positions
        q_all_per_pos = shift_q.mean(dim=-1)  # (B, L-1)
        q_all_mean = ((q_all_per_pos * mask).sum() / (mask.sum() + 1e-8)).item()
        q_taken_advantage = (q_pred_global_mean - q_all_mean) if valid_q_pred.numel() > 0 else 0.0

        # === Policy statistics ===
        policy_probs_all = F.softmax(shift_logits, dim=-1)  # (B, L-1, V)
        # Probability assigned to taken action
        taken_probs = policy_probs_all.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        valid_taken_probs = taken_probs[mask > 0]
        taken_prob_mean = valid_taken_probs.mean().item() if valid_taken_probs.numel() > 0 else 0.0
        taken_log_prob_mean = valid_taken_probs.log().mean().item() if valid_taken_probs.numel() > 0 else 0.0

        # === KL divergence between LM policy and Q-tilted policy ===
        kl_div = (lm_probs * (lm_log_probs - tilted_log_probs)).sum(dim=-1)  # (B, L-1)
        kl_div_mean = ((kl_div * mask).sum() / (mask.sum() + 1e-8)).item()

        # === Gradient norms (computed after backward) ===
        total_grad_norm = 0.0
        q_head_grad_norm = 0.0
        lm_head_grad_norm = 0.0
        transformer_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item() ** 2
                total_grad_norm += param_norm
                if 'q_head' in name:
                    q_head_grad_norm += param_norm
                elif 'lm_head' in name:
                    lm_head_grad_norm += param_norm
                else:
                    transformer_grad_norm += param_norm
        total_grad_norm = total_grad_norm ** 0.5
        q_head_grad_norm = q_head_grad_norm ** 0.5
        lm_head_grad_norm = lm_head_grad_norm ** 0.5
        transformer_grad_norm = transformer_grad_norm ** 0.5

    return {
        # Basic losses
        "loss": loss.item(),
        "lm_loss": lm_loss.item(),
        "q_loss": q_loss.item(),
        "perplexity": perplexity,

        # Position-wise arrays (numpy)
        "q_loss_by_pos": q_loss_by_pos,
        "returns_var_by_pos": returns_var_by_pos,
        "q_pred_mean_by_pos": q_pred_mean_by_pos,
        "q_pred_var_by_pos": q_pred_var_by_pos,
        "returns_mean_by_pos": returns_mean_by_pos_np,
        "lm_loss_by_pos": lm_loss_by_pos,
        "lm_entropy_by_pos": lm_entropy_by_pos,
        "top1_acc_by_pos": top1_acc_by_pos,
        "tilted_entropy_by_pos": tilted_entropy_by_pos,

        # Global Q statistics
        "q_pred_global_mean": q_pred_global_mean,
        "q_pred_global_std": q_pred_global_std,
        "q_pred_global_min": q_pred_global_min,
        "q_pred_global_max": q_pred_global_max,

        # Global returns statistics
        "returns_global_mean": returns_global_mean,
        "returns_global_std": returns_global_std,
        "returns_global_min": returns_global_min,
        "returns_global_max": returns_global_max,

        # Bootstrap statistics
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_std": bootstrap_std,

        # Reward statistics
        "rewards_mean": rewards_mean,
        "rewards_std": rewards_std,
        "rewards_nonzero_frac": rewards_nonzero_frac,

        # Q advantage (taken action Q vs average Q)
        "q_taken_advantage": q_taken_advantage,

        # Policy statistics
        "taken_prob_mean": taken_prob_mean,
        "taken_log_prob_mean": taken_log_prob_mean,
        "kl_div_lm_tilted": kl_div_mean,

        # Gradient norms
        "grad_norm_total": total_grad_norm,
        "grad_norm_q_head": q_head_grad_norm,
        "grad_norm_lm_head": lm_head_grad_norm,
        "grad_norm_transformer": transformer_grad_norm,
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


def save_checkpoint(model, optimizer, step, save_dir, tokenizer=None, args=None):
    """Save model checkpoint.

    Args:
        model: GPT2WithQ model
        optimizer: optimizer
        step: current training step
        save_dir: directory to save checkpoints
        tokenizer: optional tokenizer to save
        args: optional training args to save
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if args is not None:
        checkpoint["args"] = vars(args)

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Also save a "latest" symlink/copy for easy loading
    latest_path = os.path.join(save_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """Load a checkpoint.

    Args:
        checkpoint_path: path to checkpoint file
        model: GPT2WithQ model to load weights into
        optimizer: optional optimizer to load state into
        device: device to map tensors to

    Returns:
        step: training step the checkpoint was saved at
        args: training args if saved, else None
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    step = checkpoint.get("step", 0)
    args = checkpoint.get("args", None)

    print(f"Loaded checkpoint from step {step}")
    return step, args


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
                 text_field="text", reward_mode="zero", split="train",
                 shuffle_buffer_size=10000, seed=None):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_field = text_field
        self.reward_mode = reward_mode
        self.split = split
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def __iter__(self):
        # Load dataset in streaming mode
        ds = load_dataset(
            self.dataset_name,
            self.config_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True
        )

        # Shuffle the dataset using a buffer - this shuffles documents before tokenization
        if self.shuffle_buffer_size > 0:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)

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


def build_streaming_dataset(tokenizer, dataset="wikipedia", block_size=128, split="train",
                            shuffle_buffer_size=10000, seed=None):
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
        shuffle_buffer_size: Size of buffer for shuffling documents (0 to disable)
        seed: Random seed for reproducible shuffling

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

    shuffle_str = f", shuffle_buffer={shuffle_buffer_size}" if shuffle_buffer_size > 0 else ""
    print(f"Loading streaming dataset: {ds_name}" + (f" ({config})" if config else "") + shuffle_str)

    return StreamingTextDataset(
        dataset_name=ds_name,
        config_name=config,
        tokenizer=tokenizer,
        block_size=block_size,
        text_field=text_field,
        split=split,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed
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
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="gpt2-q-head", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    # Reward mode
    parser.add_argument("--use_lm_rewards", action="store_true",
                        help="Use log P(next_token) from reference model as rewards instead of batch rewards")
    # Checkpoint saving
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g., checkpoints/checkpoint_latest.pt)")
    args = parser.parse_args()

    # Default to steps mode for streaming datasets, epochs for synthetic
    if args.steps is None and args.epochs is None:
        if args.dataset == "synthetic":
            args.epochs = 3
        else:
            args.steps = 10000  # Default 10k steps for streaming

    # Initialize Weights & Biases
    use_wandb = args.wandb and HAS_WANDB
    if args.wandb and not HAS_WANDB:
        print("Warning: --wandb specified but wandb not installed. Run: pip install wandb")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "dataset": args.dataset,
                "steps": args.steps,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "gamma": args.gamma,
                "q_weight": args.q_weight,
                "max_len": args.max_len,
                "use_lm_rewards": args.use_lm_rewards,
                "save_interval": args.save_interval,
            }
        )
        print(f"W&B initialized: {wandb.run.url}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # Auto-scale batch size based on GPU memory if not specified
    if args.batch_size is None:
        args.batch_size = estimate_batch_size(args.device, seq_len=args.max_len)
        if use_wandb:
            wandb.config.update({"batch_size": args.batch_size})
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

    # Resume from checkpoint if specified
    resume_step = 0
    if args.resume:
        if os.path.exists(args.resume):
            resume_step, loaded_args = load_checkpoint(args.resume, model, optimizer, device=args.device)
            print(f"Resuming from step {resume_step}")
        else:
            print(f"Warning: Checkpoint not found at {args.resume}, starting from scratch")

    # Reference model (standard GPT-2, no Q-head) for comparison
    # Note: ref_model is always needed if use_lm_rewards is True
    if not args.no_reference or args.use_lm_rewards:
        ref_model = GPT2LMHeadModel.from_pretrained("gpt2")
        ref_model.resize_token_embeddings(len(tokenizer))
        ref_model.to(args.device)
        if not args.no_reference:
            ref_optimizer = torch.optim.AdamW(ref_model.parameters(), lr=args.lr)
            print("Training Q-head model and reference GPT-2 in lockstep...")
        else:
            ref_optimizer = None
            print("Training Q-head model only (using ref model for rewards)...")
    else:
        ref_model = None
        ref_optimizer = None
        print("Training Q-head model only...")

    if args.use_lm_rewards:
        print("Using log P(next_token) from reference model as rewards")
    else:
        print("Using dataset rewards (zero for streaming, random for synthetic)")

    print(f"{'='*80}")
    print(f"{'Step':>8} | {'Q-Model PPL':>12} {'Q-Model LM':>12} {'Q-Loss':>10} | {'Ref PPL':>12} {'Ref LM':>10}")
    print(f"{'='*80}")

    # Unified training loop for both streaming and epoch-based modes
    global_step = resume_step
    acc_q = {}  # Will accumulate all scalar metrics
    acc_ref = {"lm_loss": 0.0, "perplexity": 0.0}

    # Position-wise array keys (these need separate accumulation)
    POSITION_WISE_KEYS = {
        "q_loss_by_pos", "returns_var_by_pos", "q_pred_mean_by_pos", "q_pred_var_by_pos",
        "returns_mean_by_pos", "lm_loss_by_pos", "lm_entropy_by_pos", "top1_acc_by_pos",
        "tilted_entropy_by_pos"
    }
    acc_pos_arrays = {}  # Will accumulate position-wise arrays
    interval_count = 0

    if streaming or args.steps is not None:
        # Step-based training (streaming mode)
        total_steps = args.steps or 10000
        data_iter = iter(dataloader)
        pbar = tqdm(total=total_steps, initial=resume_step, desc="Training")

        while global_step < total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart iterator if dataset exhausted (shouldn't happen with streaming)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Train Q-head model
            stats_q = train_step(model, batch, optimizer, tokenizer, gamma=args.gamma,
                                 q_loss_weight=args.q_weight, device=args.device,
                                 ref_model=ref_model, use_lm_rewards=args.use_lm_rewards)

            # Accumulate scalar metrics
            for k, v in stats_q.items():
                if k not in POSITION_WISE_KEYS:
                    if k not in acc_q:
                        acc_q[k] = 0.0
                    acc_q[k] += v

            # Accumulate position-wise arrays
            for k in POSITION_WISE_KEYS:
                arr = stats_q.get(k)
                if arr is not None:
                    if k not in acc_pos_arrays:
                        acc_pos_arrays[k] = np.zeros_like(arr)
                    acc_pos_arrays[k] += arr

            # Train reference model on same batch (only if we're actually training it)
            if ref_optimizer is not None:
                stats_ref = train_step_reference(ref_model, batch, ref_optimizer, device=args.device)
                for k, v in stats_ref.items():
                    acc_ref[k] += v

            global_step += 1
            interval_count += 1
            pbar.update(1)

            # Save checkpoint at regular intervals
            if global_step % args.save_interval == 0:
                save_checkpoint(model, optimizer, global_step, args.save_dir, args=args)

            # Log diagnostics every N steps
            if global_step % args.log_interval == 0:
                q_ppl = acc_q.get("perplexity", 0.0) / interval_count
                q_lm = acc_q.get("lm_loss", 0.0) / interval_count
                q_loss = acc_q.get("q_loss", 0.0) / interval_count

                # W&B logging
                if use_wandb:
                    log_dict = {
                        "step": global_step,
                        "q_model/perplexity": q_ppl,
                        "q_model/lm_loss": q_lm,
                        "q_model/q_loss": q_loss,
                    }
                    if ref_optimizer is not None:
                        ref_ppl = acc_ref["perplexity"] / interval_count
                        ref_lm = acc_ref["lm_loss"] / interval_count
                        log_dict["ref_model/perplexity"] = ref_ppl
                        log_dict["ref_model/lm_loss"] = ref_lm
                        log_dict["delta_ppl"] = q_ppl - ref_ppl

                    # === Global scalar metrics ===
                    scalar_metrics = [
                        # Global Q statistics
                        ("q_stats/global_mean", "q_pred_global_mean"),
                        ("q_stats/global_std", "q_pred_global_std"),
                        ("q_stats/global_min", "q_pred_global_min"),
                        ("q_stats/global_max", "q_pred_global_max"),
                        # Global returns statistics
                        ("returns_stats/global_mean", "returns_global_mean"),
                        ("returns_stats/global_std", "returns_global_std"),
                        ("returns_stats/global_min", "returns_global_min"),
                        ("returns_stats/global_max", "returns_global_max"),
                        # Bootstrap statistics
                        ("bootstrap/mean", "bootstrap_mean"),
                        ("bootstrap/std", "bootstrap_std"),
                        # Reward statistics
                        ("rewards/mean", "rewards_mean"),
                        ("rewards/std", "rewards_std"),
                        ("rewards/nonzero_frac", "rewards_nonzero_frac"),
                        # Q advantage
                        ("q_stats/taken_advantage", "q_taken_advantage"),
                        # Policy statistics
                        ("policy/taken_prob_mean", "taken_prob_mean"),
                        ("policy/taken_log_prob_mean", "taken_log_prob_mean"),
                        ("policy/kl_div_lm_tilted", "kl_div_lm_tilted"),
                        # Gradient norms
                        ("gradients/total_norm", "grad_norm_total"),
                        ("gradients/q_head_norm", "grad_norm_q_head"),
                        ("gradients/lm_head_norm", "grad_norm_lm_head"),
                        ("gradients/transformer_norm", "grad_norm_transformer"),
                    ]
                    for log_key, acc_key in scalar_metrics:
                        if acc_key in acc_q:
                            log_dict[log_key] = acc_q[acc_key] / interval_count

                    # === Position-wise array logging ===
                    key_positions = [0, 5, 10, 20, 30, 50, 80, 100, 120]

                    for arr_name in POSITION_WISE_KEYS:
                        if arr_name in acc_pos_arrays:
                            avg_arr = acc_pos_arrays[arr_name] / interval_count
                            # Log key individual positions
                            for pos in key_positions:
                                if pos < len(avg_arr):
                                    log_dict[f"{arr_name}/pos_{pos:03d}"] = avg_arr[pos]
                            # Log range means
                            if len(avg_arr) >= 30:
                                log_dict[f"{arr_name}/mean_early_0_30"] = avg_arr[:30].mean()
                            if len(avg_arr) >= 90:
                                log_dict[f"{arr_name}/mean_mid_30_90"] = avg_arr[30:90].mean()
                            if len(avg_arr) > 90:
                                log_dict[f"{arr_name}/mean_late_90_end"] = avg_arr[90:].mean()

                    wandb.log(log_dict, step=global_step)

                if ref_optimizer is not None:
                    ref_ppl = acc_ref["perplexity"] / interval_count
                    ref_lm = acc_ref["lm_loss"] / interval_count
                    ppl_diff = q_ppl - ref_ppl
                    tqdm.write(f"{global_step:>8} | {q_ppl:>12.2f} {q_lm:>12.4f} {q_loss:>10.4f} | {ref_ppl:>12.2f} {ref_lm:>10.4f}  (Δppl={ppl_diff:+.2f})")
                else:
                    tqdm.write(f"{global_step:>8} | {q_ppl:>12.2f} {q_lm:>12.4f} {q_loss:>10.4f} |")
                # Reset interval accumulators
                acc_q = {}
                acc_ref = {"lm_loss": 0.0, "perplexity": 0.0}
                acc_pos_arrays = {}
                interval_count = 0

        pbar.close()
    else:
        # Epoch-based training (finite dataset)
        for epoch in range(args.epochs):
            epoch_q = {}
            epoch_ref = {"lm_loss": 0.0, "perplexity": 0.0}
            epoch_count = 0

            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
                # Train Q-head model
                stats_q = train_step(model, batch, optimizer, tokenizer, gamma=args.gamma,
                                     q_loss_weight=args.q_weight, device=args.device,
                                     ref_model=ref_model, use_lm_rewards=args.use_lm_rewards)

                # Accumulate scalar metrics
                for k, v in stats_q.items():
                    if k not in POSITION_WISE_KEYS:
                        if k not in epoch_q:
                            epoch_q[k] = 0.0
                        epoch_q[k] += v
                        if k not in acc_q:
                            acc_q[k] = 0.0
                        acc_q[k] += v

                # Accumulate position-wise arrays
                for k in POSITION_WISE_KEYS:
                    arr = stats_q.get(k)
                    if arr is not None:
                        if k not in acc_pos_arrays:
                            acc_pos_arrays[k] = np.zeros_like(arr)
                        acc_pos_arrays[k] += arr

                # Train reference model on same batch (only if we're actually training it)
                if ref_optimizer is not None:
                    stats_ref = train_step_reference(ref_model, batch, ref_optimizer, device=args.device)
                    for k, v in stats_ref.items():
                        epoch_ref[k] += v
                        acc_ref[k] += v

                global_step += 1
                epoch_count += 1
                interval_count += 1

                # Save checkpoint at regular intervals
                if global_step % args.save_interval == 0:
                    save_checkpoint(model, optimizer, global_step, args.save_dir, args=args)

                # Log diagnostics every N batches
                if global_step % args.log_interval == 0:
                    q_ppl = acc_q.get("perplexity", 0.0) / interval_count
                    q_lm = acc_q.get("lm_loss", 0.0) / interval_count
                    q_loss = acc_q.get("q_loss", 0.0) / interval_count

                    # W&B logging
                    if use_wandb:
                        log_dict = {
                            "step": global_step,
                            "epoch": epoch + 1,
                            "q_model/perplexity": q_ppl,
                            "q_model/lm_loss": q_lm,
                            "q_model/q_loss": q_loss,
                        }
                        if ref_optimizer is not None:
                            ref_ppl = acc_ref["perplexity"] / interval_count
                            ref_lm = acc_ref["lm_loss"] / interval_count
                            log_dict["ref_model/perplexity"] = ref_ppl
                            log_dict["ref_model/lm_loss"] = ref_lm
                            log_dict["delta_ppl"] = q_ppl - ref_ppl

                        # === Global scalar metrics ===
                        scalar_metrics = [
                            ("q_stats/global_mean", "q_pred_global_mean"),
                            ("q_stats/global_std", "q_pred_global_std"),
                            ("q_stats/global_min", "q_pred_global_min"),
                            ("q_stats/global_max", "q_pred_global_max"),
                            ("returns_stats/global_mean", "returns_global_mean"),
                            ("returns_stats/global_std", "returns_global_std"),
                            ("returns_stats/global_min", "returns_global_min"),
                            ("returns_stats/global_max", "returns_global_max"),
                            ("bootstrap/mean", "bootstrap_mean"),
                            ("bootstrap/std", "bootstrap_std"),
                            ("rewards/mean", "rewards_mean"),
                            ("rewards/std", "rewards_std"),
                            ("rewards/nonzero_frac", "rewards_nonzero_frac"),
                            ("q_stats/taken_advantage", "q_taken_advantage"),
                            ("policy/taken_prob_mean", "taken_prob_mean"),
                            ("policy/taken_log_prob_mean", "taken_log_prob_mean"),
                            ("policy/kl_div_lm_tilted", "kl_div_lm_tilted"),
                            ("gradients/total_norm", "grad_norm_total"),
                            ("gradients/q_head_norm", "grad_norm_q_head"),
                            ("gradients/lm_head_norm", "grad_norm_lm_head"),
                            ("gradients/transformer_norm", "grad_norm_transformer"),
                        ]
                        for log_key, acc_key in scalar_metrics:
                            if acc_key in acc_q:
                                log_dict[log_key] = acc_q[acc_key] / interval_count

                        # === Position-wise array logging ===
                        key_positions = [0, 5, 10, 20, 30, 50, 80, 100, 120]

                        for arr_name in POSITION_WISE_KEYS:
                            if arr_name in acc_pos_arrays:
                                avg_arr = acc_pos_arrays[arr_name] / interval_count
                                for pos in key_positions:
                                    if pos < len(avg_arr):
                                        log_dict[f"{arr_name}/pos_{pos:03d}"] = avg_arr[pos]
                                if len(avg_arr) >= 30:
                                    log_dict[f"{arr_name}/mean_early_0_30"] = avg_arr[:30].mean()
                                if len(avg_arr) >= 90:
                                    log_dict[f"{arr_name}/mean_mid_30_90"] = avg_arr[30:90].mean()
                                if len(avg_arr) > 90:
                                    log_dict[f"{arr_name}/mean_late_90_end"] = avg_arr[90:].mean()

                        wandb.log(log_dict, step=global_step)

                    if ref_optimizer is not None:
                        ref_ppl = acc_ref["perplexity"] / interval_count
                        ref_lm = acc_ref["lm_loss"] / interval_count
                        ppl_diff = q_ppl - ref_ppl
                        tqdm.write(f"{global_step:>8} | {q_ppl:>12.2f} {q_lm:>12.4f} {q_loss:>10.4f} | {ref_ppl:>12.2f} {ref_lm:>10.4f}  (Δppl={ppl_diff:+.2f})")
                    else:
                        tqdm.write(f"{global_step:>8} | {q_ppl:>12.2f} {q_lm:>12.4f} {q_loss:>10.4f} |")
                    acc_q = {}
                    acc_ref = {"lm_loss": 0.0, "perplexity": 0.0}
                    acc_pos_arrays = {}
                    interval_count = 0

            # Epoch summary
            print(f"{'-'*80}")
            q_ppl = epoch_q.get("perplexity", 0.0) / epoch_count
            q_lm = epoch_q.get("lm_loss", 0.0) / epoch_count
            q_loss = epoch_q.get("q_loss", 0.0) / epoch_count
            if ref_optimizer is not None:
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

    # Save final checkpoint
    save_checkpoint(model, optimizer, global_step, args.save_dir, args=args)
    print(f"\nTraining complete. Final checkpoint saved at step {global_step}")

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
            if use_wandb:
                wandb.log({"diagnostics_plot": wandb.Image(args.plot_path)})

    # Finish W&B run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
