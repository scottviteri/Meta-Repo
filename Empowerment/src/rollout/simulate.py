"""Rollout simulation."""

import numpy as np

from src.envs.base import DiscreteEnv
from src.policies.base import OuterPolicy


def sample_transition(probs: dict[int, float], rng: np.random.Generator) -> int:
    states = list(probs.keys())
    p = [probs[s] for s in states]
    return int(rng.choice(states, p=p))


def rollout(
    env: DiscreteEnv,
    policy: OuterPolicy,
    horizon_T: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    """Run one episode, returning list of (state, action, next_state) tuples."""
    rng = np.random.default_rng(seed)
    s = env.initial_state()
    traj = []

    for _ in range(horizon_T):
        a = policy.sample_action(s, rng)
        s_next = sample_transition(env.transition_prob(s, a), rng)
        traj.append((s, a, s_next))
        s = s_next
        if env.is_absorbing(s):
            break

    return traj
