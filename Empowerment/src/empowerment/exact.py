"""Exact tabular empowerment computation."""

from src.envs.base import DiscreteEnv
from src.empowerment.nstep_model import build_nstep_channel
from src.empowerment.blahut_arimoto import blahut_arimoto


def exact_empowerment(env: DiscreteEnv, s: int, horizon: int) -> float:
    """Compute exact n-step empowerment for state s."""
    if env.is_absorbing(s):
        return 0.0

    channel = build_nstep_channel(env, s, horizon)
    if channel.size == 0:
        return 0.0

    return blahut_arimoto(channel)


def compute_empowerment_table(
    env: DiscreteEnv, horizon: int
) -> dict[int, float]:
    """Compute empowerment for all states."""
    table = {}
    for s in env.states:
        table[s] = exact_empowerment(env, s, horizon)
    return table
