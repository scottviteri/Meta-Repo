"""Softmax / Boltzmann outer policy."""

import math

from src.envs.base import DiscreteEnv
from src.policies.base import OuterPolicy


class SoftmaxEmpowermentPolicy(OuterPolicy):
    def __init__(
        self,
        env: DiscreteEnv,
        emp_table: dict[int, float],
        temperature: float = 1.0,
        normalize: str = "none",
    ):
        super().__init__(env, emp_table)
        self.temperature = temperature
        self.normalize = normalize

    def action_probs(self, s: int) -> dict[int, float]:
        scores = self._scores(s)
        if not scores:
            return {}

        vals = list(scores.values())

        if self.normalize == "zscore" and len(vals) > 1:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            if std > 1e-12:
                scores = {a: (v - mean) / std for a, v in scores.items()}
            else:
                scores = {a: 0.0 for a in scores}
        elif self.normalize == "max_subtract":
            m = max(vals)
            scores = {a: v - m for a, v in scores.items()}

        m = max(scores.values())
        weights = {}
        for a, v in scores.items():
            weights[a] = math.exp((v - m) / self.temperature)
        z = sum(weights.values())
        return {a: w / z for a, w in weights.items()}
