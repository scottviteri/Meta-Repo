from abc import ABC, abstractmethod


class DiscreteEnv(ABC):
    states: list[int]
    actions: dict[int, list[int]]

    @abstractmethod
    def transition_prob(self, s: int, a: int) -> dict[int, float]:
        ...

    @abstractmethod
    def is_absorbing(self, s: int) -> bool:
        ...

    @abstractmethod
    def initial_state(self) -> int:
        ...

    def successors(self, s: int, a: int) -> list[int]:
        return list(self.transition_prob(s, a).keys())
