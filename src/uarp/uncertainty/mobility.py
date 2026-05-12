"""Device mobility traces — IMPROVEMENTS.md 一.1.

Provides time-varying distances from the workflow's mobile device to each edge
node. ``Topology`` keeps a ``mobility: MobilityTrace | None`` field; cost
formulas call ``topology.distance_at(t, k)`` which falls back to the constant
``topology.distances[k]`` when no trace is attached.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MobilityTrace:
    """Piecewise-linear distance trace.

    ``times`` is a strictly increasing 1-D array of length T (seconds, or any
    consistent time unit). ``distances`` is shape ``(T, N)`` giving the
    device-to-node distance at each sample. ``distance_at(t, k)`` linearly
    interpolates between samples and clamps outside the bounds.
    """

    times: np.ndarray
    distances: np.ndarray

    def __post_init__(self) -> None:
        if self.distances.ndim != 2:
            raise ValueError("distances must be 2-D (T, N)")
        if self.distances.shape[0] != len(self.times):
            raise ValueError(
                f"times length {len(self.times)} != distances.shape[0] "
                f"{self.distances.shape[0]}"
            )

    @property
    def N(self) -> int:
        return int(self.distances.shape[1])

    def distance_at(self, t: float, k: int) -> float:
        ts = self.times
        if t <= ts[0]:
            return float(self.distances[0, k])
        if t >= ts[-1]:
            return float(self.distances[-1, k])
        idx = int(np.searchsorted(ts, t))
        t0, t1 = float(ts[idx - 1]), float(ts[idx])
        d0, d1 = float(self.distances[idx - 1, k]), float(self.distances[idx, k])
        frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return d0 + frac * (d1 - d0)


def linear_walk(
    base_distances: np.ndarray,
    velocity: float,
    T_horizon: float,
    *,
    n_samples: int = 64,
    seed: int = 0,
    min_distance: float = 1.0,
) -> MobilityTrace:
    """Constant-velocity walk: per-node signed direction (away/toward).

    For each node ``k``, draws a direction ``d_k ∈ {-1, +1}`` and computes
    ``distance(t, k) = max(base[k] + d_k · velocity · t, min_distance)``.
    Capturing a mix of approaching and receding nodes — the simplest mobility
    model that already breaks the constant-distance assumption.
    """
    rng = np.random.default_rng(seed)
    N = len(base_distances)
    directions = rng.choice([-1.0, 1.0], size=N)
    times = np.linspace(0.0, float(T_horizon), n_samples)
    raw = base_distances[None, :] + (times[:, None] * float(velocity) * directions[None, :])
    distances = np.maximum(raw, float(min_distance))
    return MobilityTrace(times=times, distances=distances)


def random_waypoint(
    base_distances: np.ndarray,
    velocity: float,
    T_horizon: float,
    *,
    n_samples: int = 64,
    pause_prob: float = 0.1,
    seed: int = 0,
    min_distance: float = 1.0,
) -> MobilityTrace:
    """Random-Waypoint Mobility (RWP, Camp et al. 2002), distance-space variant.

    At each sample step the device picks a new direction in {-1, 0, +1} per
    node independently (0 ≡ pause). Distance evolves piecewise. Distances
    are clamped to ``min_distance`` to avoid degenerate zero-distance.
    """
    rng = np.random.default_rng(seed)
    N = len(base_distances)
    times = np.linspace(0.0, float(T_horizon), n_samples)
    distances = np.empty((n_samples, N))
    distances[0] = base_distances
    for step in range(1, n_samples):
        dt = times[step] - times[step - 1]
        moves = rng.choice([-1.0, 0.0, 1.0], size=N, p=[(1 - pause_prob) / 2, pause_prob, (1 - pause_prob) / 2])
        distances[step] = np.maximum(distances[step - 1] + moves * velocity * dt, min_distance)
    return MobilityTrace(times=times, distances=distances)
