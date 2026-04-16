"""Build the n-step action-sequence to final-state channel."""

import itertools

import numpy as np

from src.envs.base import DiscreteEnv


def build_nstep_channel(
    env: DiscreteEnv, s: int, horizon: int
) -> np.ndarray:
    """Build channel matrix P(s_n | s, a_{0:n-1}).

    Returns:
        (num_action_sequences, num_states) channel matrix.
        Returns empty array if no action sequences are available.
    """
    actions_at_s = env.actions[s]
    if not actions_at_s or horizon == 0:
        return np.zeros((0, 0))

    all_action_seqs = list(itertools.product(actions_at_s, repeat=horizon))
    n_seqs = len(all_action_seqs)
    n_states = len(env.states)
    state_to_idx = {s: i for i, s in enumerate(env.states)}

    channel = np.zeros((n_seqs, n_states))

    for seq_idx, action_seq in enumerate(all_action_seqs):
        # Start with deterministic state s
        dist = {s: 1.0}

        for a in action_seq:
            next_dist: dict[int, float] = {}
            for cur_state, cur_prob in dist.items():
                if cur_prob == 0:
                    continue
                trans = env.transition_prob(cur_state, a)
                for ns, tp in trans.items():
                    next_dist[ns] = next_dist.get(ns, 0.0) + cur_prob * tp
            dist = next_dist

        for final_state, prob in dist.items():
            channel[seq_idx, state_to_idx[final_state]] = prob

    return channel
