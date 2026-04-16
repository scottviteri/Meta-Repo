"""Greedy (argmax) outer policy."""

from src.policies.base import OuterPolicy


class GreedyEmpowermentPolicy(OuterPolicy):
    def action_probs(self, s: int) -> dict[int, float]:
        scores = self._scores(s)
        if not scores:
            return {}
        max_score = max(scores.values())
        best = [a for a, v in scores.items() if abs(v - max_score) < 1e-12]
        n_best = len(best)
        return {a: (1.0 / n_best if a in best else 0.0) for a in scores}
