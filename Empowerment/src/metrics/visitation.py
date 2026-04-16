"""State visitation metrics."""

import math
from collections import Counter


def visitation_entropy(trajectory: list[tuple[int, int, int]]) -> float:
    """Shannon entropy of state visitation distribution."""
    states = [s for s, _, _ in trajectory]
    if not states:
        return 0.0
    counts = Counter(states)
    total = len(states)
    return -sum(
        (c / total) * math.log(c / total) for c in counts.values() if c > 0
    )


def distinct_states_visited(trajectory: list[tuple[int, int, int]]) -> int:
    return len({s for s, _, _ in trajectory})


def mean_empowerment_visited(
    trajectory: list[tuple[int, int, int]], emp_table: dict[int, float]
) -> float:
    if not trajectory:
        return 0.0
    return sum(emp_table.get(s, 0.0) for s, _, _ in trajectory) / len(trajectory)


def fraction_reaching_states(
    trajectories: list[list[tuple[int, int, int]]],
    target_states: set[int],
) -> float:
    """Fraction of episodes that visit any target state."""
    if not trajectories:
        return 0.0
    reached = 0
    for traj in trajectories:
        visited = {s for s, _, _ in traj} | {sn for _, _, sn in traj}
        if visited & target_states:
            reached += 1
    return reached / len(trajectories)
