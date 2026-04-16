"""Epsilon-greedy outer policy."""

from src.policies.base import OuterPolicy
from src.envs.base import DiscreteEnv


class EpsilonGreedyEmpowermentPolicy(OuterPolicy):
    def __init__(
        self,
        env: DiscreteEnv,
        emp_table: dict[int, float],
        epsilon: float = 0.1,
    ):
        super().__init__(env, emp_table)
        self.epsilon = epsilon

    def action_probs(self, s: int) -> dict[int, float]:
        scores = self._scores(s)
        if not scores:
            return {}
        n_actions = len(scores)
        max_score = max(scores.values())
        best = [a for a, v in scores.items() if abs(v - max_score) < 1e-12]
        n_best = len(best)

        probs = {}
        for a in scores:
            if a in best:
                probs[a] = (1.0 - self.epsilon) / n_best + self.epsilon / n_actions
            else:
                probs[a] = self.epsilon / n_actions
        return probs
