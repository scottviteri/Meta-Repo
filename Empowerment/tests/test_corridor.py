"""Tests for corridor environment."""

import pytest

from src.envs.corridor import CorridorEnv


def test_state_count():
    env = CorridorEnv()
    assert len(env.states) > 0


def test_transition_sums_to_one():
    env = CorridorEnv()
    for s in env.states:
        for a in env.actions[s]:
            probs = env.transition_prob(s, a)
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-10, f"State {s}, action {a}: sum={total}"


def test_deterministic():
    env = CorridorEnv()
    for s in env.states:
        for a in env.actions[s]:
            probs = env.transition_prob(s, a)
            assert len(probs) == 1, "Corridor should be deterministic"


def test_no_absorbing():
    env = CorridorEnv()
    for s in env.states:
        assert not env.is_absorbing(s)


def test_start_in_local_basin():
    env = CorridorEnv()
    start = env.initial_state()
    assert start in env.local_basin_states


def test_basins_nonempty():
    env = CorridorEnv()
    assert len(env.local_basin_states) > 0
    assert len(env.far_basin_states) > 0


def test_basins_disjoint():
    env = CorridorEnv()
    assert env.local_basin_states & env.far_basin_states == set()
