"""Tests for outer policies."""

import math

import numpy as np
import pytest

from src.envs.corridor import CorridorEnv
from src.empowerment.exact import compute_empowerment_table
from src.policies import (
    GreedyEmpowermentPolicy,
    SoftmaxEmpowermentPolicy,
    EpsilonGreedyEmpowermentPolicy,
)


@pytest.fixture(scope="module")
def env_and_table():
    env = CorridorEnv()
    table = compute_empowerment_table(env, horizon=2)
    return env, table


def test_greedy_sums_to_one(env_and_table):
    env, table = env_and_table
    policy = GreedyEmpowermentPolicy(env, table)
    for s in env.states:
        probs = policy.action_probs(s)
        assert abs(sum(probs.values()) - 1.0) < 1e-10


def test_softmax_sums_to_one(env_and_table):
    env, table = env_and_table
    for tau in [0.01, 0.1, 1.0, 10.0]:
        policy = SoftmaxEmpowermentPolicy(env, table, temperature=tau)
        for s in env.states:
            probs = policy.action_probs(s)
            assert abs(sum(probs.values()) - 1.0) < 1e-10


def test_softmax_approaches_greedy(env_and_table):
    env, table = env_and_table
    greedy = GreedyEmpowermentPolicy(env, table)
    soft = SoftmaxEmpowermentPolicy(env, table, temperature=0.001)
    for s in env.states[:10]:
        gp = greedy.action_probs(s)
        sp = soft.action_probs(s)
        for a in gp:
            assert abs(gp[a] - sp[a]) < 0.05


def test_softmax_approaches_uniform(env_and_table):
    env, table = env_and_table
    policy = SoftmaxEmpowermentPolicy(env, table, temperature=1000.0)
    for s in env.states[:10]:
        probs = policy.action_probs(s)
        n = len(probs)
        for p in probs.values():
            assert abs(p - 1.0 / n) < 0.01


def test_epsilon_greedy_sums_to_one(env_and_table):
    env, table = env_and_table
    policy = EpsilonGreedyEmpowermentPolicy(env, table, epsilon=0.2)
    for s in env.states:
        probs = policy.action_probs(s)
        assert abs(sum(probs.values()) - 1.0) < 1e-10
