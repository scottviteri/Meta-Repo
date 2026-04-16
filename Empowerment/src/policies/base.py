"""Base class for outer policies."""

from abc import ABC, abstractmethod

import numpy as np

from src.envs.base import DiscreteEnv
from src.empowerment.score import empowerment_action_scores


class OuterPolicy(ABC):
    def __init__(self, env: DiscreteEnv, emp_table: dict[int, float]):
        self.env = env
        self.emp_table = emp_table

    def _scores(self, s: int) -> dict[int, float]:
        return empowerment_action_scores(self.env, self.emp_table, s)

    @abstractmethod
    def action_probs(self, s: int) -> dict[int, float]:
        ...

    def sample_action(self, s: int, rng: np.random.Generator) -> int:
        probs = self.action_probs(s)
        actions = list(probs.keys())
        p = [probs[a] for a in actions]
        return int(rng.choice(actions, p=p))
