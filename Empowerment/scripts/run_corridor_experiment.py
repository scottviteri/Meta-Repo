#!/usr/bin/env python3
"""Run the corridor experiment: argmax vs softmax temperature sweep.

This is Milestone 1 — the minimal experiment that demonstrates
whether softmax outer control helps escape local empowerment maxima.
"""

import json
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs.corridor import CorridorEnv
from src.empowerment.exact import compute_empowerment_table
from src.policies import (
    GreedyEmpowermentPolicy,
    SoftmaxEmpowermentPolicy,
    EpsilonGreedyEmpowermentPolicy,
)
from src.rollout.simulate import rollout
from src.metrics.visitation import (
    visitation_entropy,
    distinct_states_visited,
    mean_empowerment_visited,
    fraction_reaching_states,
)


def run_experiment(
    horizon: int = 3,
    rollout_length: int = 200,
    n_seeds: int = 100,
    temperatures: list[float] | None = None,
    epsilon_values: list[float] | None = None,
):
    if temperatures is None:
        temperatures = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    if epsilon_values is None:
        epsilon_values = [0.1, 0.3]

    env = CorridorEnv()

    print(f"Environment: {len(env.states)} states")
    print(f"Local basin: {len(env.local_basin_states)} states")
    print(f"Far basin: {len(env.far_basin_states)} states")
    print(f"Start state: {env.initial_state()} at position {env.state_pos(env.initial_state())}")
    print()

    print(f"Computing empowerment table (horizon={horizon})...")
    emp_table = compute_empowerment_table(env, horizon)

    local_emp = np.mean([emp_table[s] for s in env.local_basin_states])
    far_emp = np.mean([emp_table[s] for s in env.far_basin_states])
    print(f"  Mean empowerment — local basin: {local_emp:.3f}, far basin: {far_emp:.3f}")
    print()

    results = []

    # Greedy
    print("Running greedy policy...")
    policy = GreedyEmpowermentPolicy(env, emp_table)
    trajs = [rollout(env, policy, rollout_length, seed=s) for s in range(n_seeds)]
    metrics = compute_metrics(trajs, env, emp_table, "greedy", tau=None, epsilon=None)
    results.append(metrics)
    print_metrics(metrics)

    # Softmax sweep
    for tau in temperatures:
        print(f"Running softmax (tau={tau})...")
        policy = SoftmaxEmpowermentPolicy(env, emp_table, temperature=tau)
        trajs = [rollout(env, policy, rollout_length, seed=s) for s in range(n_seeds)]
        metrics = compute_metrics(trajs, env, emp_table, "softmax", tau=tau, epsilon=None)
        results.append(metrics)
        print_metrics(metrics)

    # Epsilon-greedy
    for eps in epsilon_values:
        print(f"Running epsilon-greedy (eps={eps})...")
        policy = EpsilonGreedyEmpowermentPolicy(env, emp_table, epsilon=eps)
        trajs = [rollout(env, policy, rollout_length, seed=s) for s in range(n_seeds)]
        metrics = compute_metrics(trajs, env, emp_table, "epsilon_greedy", tau=None, epsilon=eps)
        results.append(metrics)
        print_metrics(metrics)

    output_dir = Path("results/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"corridor_h{horizon}_T{rollout_length}_n{n_seeds}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print_summary_table(results)

    return results


def compute_metrics(trajs, env, emp_table, policy_name, tau, epsilon):
    entropies = [visitation_entropy(t) for t in trajs]
    distinct = [distinct_states_visited(t) for t in trajs]
    mean_emp = [mean_empowerment_visited(t, emp_table) for t in trajs]
    frac_far = fraction_reaching_states(trajs, env.far_basin_states)

    return {
        "policy": policy_name,
        "temperature": tau,
        "epsilon": epsilon,
        "frac_reaching_far_basin": frac_far,
        "mean_visitation_entropy": float(np.mean(entropies)),
        "std_visitation_entropy": float(np.std(entropies)),
        "mean_distinct_states": float(np.mean(distinct)),
        "mean_empowerment_visited": float(np.mean(mean_emp)),
        "std_empowerment_visited": float(np.std(mean_emp)),
    }


def print_metrics(m):
    label = m["policy"]
    if m["temperature"] is not None:
        label += f" (tau={m['temperature']})"
    if m["epsilon"] is not None:
        label += f" (eps={m['epsilon']})"
    print(f"  {label}: far_basin={m['frac_reaching_far_basin']:.2f}, "
          f"entropy={m['mean_visitation_entropy']:.3f}, "
          f"distinct={m['mean_distinct_states']:.1f}, "
          f"emp={m['mean_empowerment_visited']:.3f}")


def print_summary_table(results):
    print("\n" + "=" * 90)
    print(f"{'Policy':<30} {'Far Basin%':>10} {'Entropy':>10} {'Distinct':>10} {'Mean Emp':>10}")
    print("-" * 90)
    for m in results:
        label = m["policy"]
        if m["temperature"] is not None:
            label += f" (tau={m['temperature']})"
        if m["epsilon"] is not None:
            label += f" (eps={m['epsilon']})"
        print(f"{label:<30} {m['frac_reaching_far_basin']:>10.2f} "
              f"{m['mean_visitation_entropy']:>10.3f} "
              f"{m['mean_distinct_states']:>10.1f} "
              f"{m['mean_empowerment_visited']:>10.3f}")
    print("=" * 90)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--rollout-length", type=int, default=200)
    parser.add_argument("--n-seeds", type=int, default=100)
    args = parser.parse_args()

    run_experiment(
        horizon=args.horizon,
        rollout_length=args.rollout_length,
        n_seeds=args.n_seeds,
    )
