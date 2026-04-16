"""Corridor-to-better-basin gridworld.

Layout (0-indexed row, col):

    ##############
    #LLL#........#
    #LLL#.######.#
    #LLL..######.#
    #LLL#.######.#
    #LLL#........#
    ##############

L = local basin (3x4 open room, rows 1-4, cols 1-4 minus wall column)
. = corridor + far basin
Start = entrance of corridor at (3, 4)

Local basin: 4x3 room → moderate empowerment
Corridor: 1-wide, 2 cells long → low empowerment
Far basin: 5x7 open room → high empowerment

We encode positions as integer state IDs.
Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
"""

import numpy as np

from src.envs.base import DiscreteEnv

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}
_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


class CorridorEnv(DiscreteEnv):
    """Deterministic gridworld with local and far empowerment basins."""

    def __init__(self):
        self._build_grid()
        self._pos_to_id = {}
        self._id_to_pos = {}
        idx = 0
        for r in range(self.nrows):
            for c in range(self.ncols):
                if self.grid[r, c] == 0:
                    self._pos_to_id[(r, c)] = idx
                    self._id_to_pos[idx] = (r, c)
                    idx += 1

        self.states = list(range(idx))
        self.actions = {s: [UP, DOWN, LEFT, RIGHT] for s in self.states}
        self._start = self._pos_to_id[(4, 1)]

        self.local_basin_states = set()
        self.far_basin_states = set()
        for sid, (r, c) in self._id_to_pos.items():
            if 1 <= r <= 7 and 1 <= c <= 4:
                self.local_basin_states.add(sid)
            if 1 <= r <= 7 and 8 <= c <= 14:
                self.far_basin_states.add(sid)

    def _build_grid(self):
        self.nrows = 9
        self.ncols = 16
        self.grid = np.ones((self.nrows, self.ncols), dtype=int)

        # Local basin: rows 1-7, cols 1-4
        for r in range(1, 8):
            for c in range(1, 5):
                self.grid[r, c] = 0

        # Corridor: row 4, cols 5-6
        self.grid[4, 5] = 0
        self.grid[4, 6] = 0
        self.grid[4, 7] = 0

        # Far basin: rows 1-7, cols 8-14
        for r in range(1, 8):
            for c in range(8, 15):
                self.grid[r, c] = 0

    def transition_prob(self, s: int, a: int) -> dict[int, float]:
        r, c = self._id_to_pos[s]
        dr, dc = _DELTAS[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.nrows and 0 <= nc < self.ncols and self.grid[nr, nc] == 0:
            return {self._pos_to_id[(nr, nc)]: 1.0}
        return {s: 1.0}

    def is_absorbing(self, s: int) -> bool:
        return False

    def initial_state(self) -> int:
        return self._start

    def state_pos(self, s: int) -> tuple[int, int]:
        return self._id_to_pos[s]

    def pos_to_state(self, r: int, c: int) -> int | None:
        return self._pos_to_id.get((r, c))

    def render(self, trajectory_states: list[int] | None = None) -> str:
        visit_counts = {}
        if trajectory_states:
            for s in trajectory_states:
                visit_counts[s] = visit_counts.get(s, 0) + 1

        lines = []
        for r in range(self.nrows):
            row = []
            for c in range(self.ncols):
                if self.grid[r, c] == 1:
                    row.append("#")
                else:
                    sid = self._pos_to_id.get((r, c))
                    if sid is not None and sid in visit_counts:
                        row.append("*")
                    elif sid == self._start:
                        row.append("S")
                    else:
                        row.append(".")
            lines.append("".join(row))
        return "\n".join(lines)
