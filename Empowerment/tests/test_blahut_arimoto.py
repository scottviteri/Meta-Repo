"""Tests for Blahut-Arimoto channel capacity computation."""

import math

import numpy as np
import pytest

from src.empowerment.blahut_arimoto import blahut_arimoto


def test_identity_channel():
    """Binary symmetric channel with p=0 is identity, capacity = log(n)."""
    n = 4
    channel = np.eye(n)
    cap = blahut_arimoto(channel)
    assert abs(cap - math.log(n)) < 1e-4


def test_single_input():
    channel = np.array([[0.5, 0.5]])
    assert blahut_arimoto(channel) < 1e-6


def test_useless_channel():
    """All inputs map to same output distribution → capacity 0."""
    channel = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    assert blahut_arimoto(channel) < 1e-6


def test_binary_symmetric_channel():
    """BSC with crossover p has known capacity."""
    p = 0.1
    channel = np.array([[1 - p, p], [p, 1 - p]])
    h_p = -p * math.log(p) - (1 - p) * math.log(1 - p)
    expected = math.log(2) - h_p
    cap = blahut_arimoto(channel)
    assert abs(cap - expected) < 1e-4


def test_deterministic_channel():
    """Deterministic mapping — capacity = log(num distinct outputs)."""
    channel = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])
    cap = blahut_arimoto(channel)
    assert abs(cap - math.log(3)) < 1e-4
