"""Empowerment-derived action scores."""

from src.envs.base import DiscreteEnv


def empowerment_action_scores(
    env: DiscreteEnv, emp_table: dict[int, float], s: int
) -> dict[int, float]:
    """Compute Q_emp(s, a) for all actions available at s."""
    scores = {}
    for a in env.actions[s]:
        probs = env.transition_prob(s, a)
        scores[a] = sum(p * emp_table.get(s_next, 0.0) for s_next, p in probs.items())
    return scores
