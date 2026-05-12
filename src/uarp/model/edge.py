"""Edge node + topology model — paper §2.2.1 and Table 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EdgeNode:
    idx: int
    capacity: float  # B_k — processing capability (MHz)
    ce: float  # ce(z_k) — execution energy per unit data (kW·s / size-unit)
    available: bool = True  # toggled by uncertainty events


@dataclass
class Topology:
    """Wireless devices ↔ edge nodes ↔ controllers.

    The mobile device for a workflow sees `distances[k]` to node k.
    Controllers exchange synchronisation messages described by `h` and `cs`.
    When ``mobility`` is set (一.1), ``distance_at(t, k)`` returns the
    time-varying distance instead of the constant ``distances[k]``.
    """

    nodes: list[EdgeNode]
    BA: float  # bandwidth (mbps) — paper Table 2 default 1000
    distances: np.ndarray  # shape (N,) — d_{m,i} per edge node
    ct_per_distance: float  # transmission energy per (size · distance)
    h: np.ndarray  # shape (U, U) — controller sync indicator
    cs: np.ndarray  # shape (U, U) — controller sync energy
    mobility: object | None = None  # MobilityTrace; None = static distances

    @property
    def N(self) -> int:
        return len(self.nodes)

    def node(self, k: int) -> EdgeNode:
        return self.nodes[k]

    def distance_at(self, t: float, k: int) -> float:
        # Fall back to the static distance for nodes the trace doesn't cover —
        # e.g. nodes added mid-execution by a ``new_node_join`` event.
        if self.mobility is None or k >= getattr(self.mobility, "N", 0):
            return float(self.distances[k])
        return float(self.mobility.distance_at(t, k))


def make_homogeneous_topology(
    N: int = 20,
    capacity: float = 2000.0,
    ce: float = 0.05,
    BA: float = 1000.0,
    distance_range: tuple[float, float] = (10.0, 100.0),
    ct_per_distance: float = 1e-5,
    n_controllers: int = 2,
    cs_value: float = 1e-3,
    seed: int | None = None,
) -> Topology:
    """Default topology matching paper Table 2 (N=20, B=2000 MHz, BA=1000 mbps)."""
    rng = np.random.default_rng(seed)
    nodes = [EdgeNode(idx=k, capacity=capacity, ce=ce) for k in range(N)]
    distances = rng.uniform(distance_range[0], distance_range[1], size=N)
    h = np.ones((n_controllers, n_controllers), dtype=int) - np.eye(n_controllers, dtype=int)
    cs = np.full((n_controllers, n_controllers), cs_value)
    np.fill_diagonal(cs, 0.0)
    return Topology(
        nodes=nodes,
        BA=BA,
        distances=distances,
        ct_per_distance=ct_per_distance,
        h=h,
        cs=cs,
    )


def make_heterogeneous_topology(
    N: int = 20,
    capacity_range: tuple[float, float] = (1500.0, 2500.0),
    ce_range: tuple[float, float] = (0.04, 0.07),
    BA: float = 1000.0,
    distance_range: tuple[float, float] = (10.0, 100.0),
    ct_per_distance: float = 1e-5,
    n_controllers: int = 2,
    cs_value: float = 1e-3,
    seed: int | None = None,
) -> Topology:
    """Heterogeneous nodes — induces a real WT vs CM trade-off.

    The paper does not detail per-node parameter variance. We make capacity,
    execution-energy coefficient, and distance all vary across nodes so that
    "fast" and "energy-efficient" nodes are different — which is necessary
    for NSGA-III to find a non-trivial Pareto front.
    """
    rng = np.random.default_rng(seed)
    capacities = rng.uniform(capacity_range[0], capacity_range[1], size=N)
    ces = rng.uniform(ce_range[0], ce_range[1], size=N)
    nodes = [
        EdgeNode(idx=k, capacity=float(capacities[k]), ce=float(ces[k]))
        for k in range(N)
    ]
    distances = rng.uniform(distance_range[0], distance_range[1], size=N)
    h = np.ones((n_controllers, n_controllers), dtype=int) - np.eye(n_controllers, dtype=int)
    cs = np.full((n_controllers, n_controllers), cs_value)
    np.fill_diagonal(cs, 0.0)
    return Topology(
        nodes=nodes,
        BA=BA,
        distances=distances,
        ct_per_distance=ct_per_distance,
        h=h,
        cs=cs,
    )
