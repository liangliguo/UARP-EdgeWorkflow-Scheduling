"""Schedule decision variable — paper §2.2.1 Eq. (3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Schedule:
    """Maps task index -> edge node index. Length M, values in [0, N-1]."""

    assignment: np.ndarray  # shape (M,) int

    @classmethod
    def from_list(cls, xs: list[int]) -> "Schedule":
        return cls(assignment=np.asarray(xs, dtype=int))

    def node_of(self, task_idx: int) -> int:
        return int(self.assignment[task_idx])

    def x(self, task_idx: int, node_idx: int) -> int:
        """Indicator x_{m,i,k} ∈ {0, 1} (Eq. 3)."""
        return 1 if self.node_of(task_idx) == node_idx else 0

    def __len__(self) -> int:
        return len(self.assignment)
