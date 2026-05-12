"""Uncertainty events injected into edge nodes (paper §3.3).

Three event types:
- performance_degradation: node B_k is multiplied by a factor in [0.3, 0.7]
- service_failure: node becomes unavailable for the rest of the workflow
- new_node_join: a new edge node is added to the topology

Events are described as plain dataclasses + a helper that applies them in-place.
A seeded EventGenerator produces a reproducible event sequence.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np

from uarp.model import EdgeNode, Topology


EventKind = Literal["performance_degradation", "service_failure", "new_node_join"]


@dataclass
class Event:
    kind: EventKind
    node_idx: int  # for join events: index of the new node (== len(nodes) before append)
    factor: float = 1.0  # only used by performance_degradation
    capacity: float = 0.0  # only used by new_node_join
    ce: float = 0.0  # only used by new_node_join
    distance: float = 0.0  # only used by new_node_join
    time: float = 0.0  # arrival timestamp (used by NHPP scheduling — see 一.2)

    def apply(self, topo: Topology) -> Topology:
        """Return a *new* Topology with the event applied (no aliasing).

        service_failure is modelled as severe performance degradation
        (capacity * 0.05) rather than full unavailability so that the
        Benchmark method's "exceed the deadline" semantics from paper §4.1(1)
        emerge naturally instead of producing infinite completion times.
        """
        new_topo = copy.deepcopy(topo)
        if self.kind == "performance_degradation":
            new_topo.nodes[self.node_idx].capacity *= self.factor
        elif self.kind == "service_failure":
            new_topo.nodes[self.node_idx].capacity *= 0.05
        elif self.kind == "new_node_join":
            new_topo.nodes.append(
                EdgeNode(idx=self.node_idx, capacity=self.capacity, ce=self.ce)
            )
            new_topo.distances = np.append(new_topo.distances, self.distance)
        return new_topo


def generate_events(
    topo: Topology,
    n_events: int = 1,
    *,
    seed: int = 0,
    distance_range: tuple[float, float] = (10.0, 100.0),
) -> list[Event]:
    """Sample a seeded sequence of events affecting the given topology."""
    rng = np.random.default_rng(seed)
    kinds: list[EventKind] = ["performance_degradation", "service_failure", "new_node_join"]
    events: list[Event] = []
    next_join_idx = topo.N
    for _ in range(n_events):
        kind = kinds[rng.integers(0, 3)]
        if kind == "performance_degradation":
            events.append(
                Event(
                    kind=kind,
                    node_idx=int(rng.integers(0, topo.N)),
                    factor=float(rng.uniform(0.3, 0.7)),
                )
            )
        elif kind == "service_failure":
            events.append(Event(kind=kind, node_idx=int(rng.integers(0, topo.N))))
        else:  # new_node_join
            events.append(
                Event(
                    kind=kind,
                    node_idx=next_join_idx,
                    capacity=float(rng.uniform(1500.0, 2500.0)),
                    ce=0.05,
                    distance=float(rng.uniform(*distance_range)),
                )
            )
            next_join_idx += 1
    return events


def apply_events(topo: Topology, events: list[Event]) -> Topology:
    """Apply a sequence of events in order, returning the resulting topology.

    Events are sorted by ``time`` before being applied so callers can pass an
    unsorted list. Events with default ``time=0.0`` preserve insertion order
    (Python's sort is stable) — that matches the pre-一.2 behaviour.
    """
    out = topo
    for ev in sorted(events, key=lambda e: e.time):
        out = ev.apply(out)
    return out


def generate_events_nhpp(
    topo: Topology,
    rate: float,
    T_horizon: float,
    *,
    seed: int = 0,
    distance_range: tuple[float, float] = (10.0, 100.0),
    rate_fn=None,
) -> list[Event]:
    """Sample an event sequence from a (non)homogeneous Poisson process.

    Implements thinning (Lewis & Shedler 1979): an upper-bound homogeneous
    Poisson process at intensity ``rate`` produces candidate timestamps,
    each kept with probability ``rate_fn(t) / rate`` if ``rate_fn`` is given.
    Returns events sorted by ``time``. Use the simpler ``generate_events``
    helper if you only need a fixed count.
    """
    rng = np.random.default_rng(seed)
    kinds: list[EventKind] = ["performance_degradation", "service_failure", "new_node_join"]
    events: list[Event] = []
    next_join_idx = topo.N
    t = 0.0
    while True:
        # Exp(rate) inter-arrival; rng.exponential takes a scale=1/rate.
        t += float(rng.exponential(1.0 / rate)) if rate > 0 else float("inf")
        if t >= T_horizon:
            break
        if rate_fn is not None and rng.random() > rate_fn(t) / rate:
            continue
        kind = kinds[rng.integers(0, 3)]
        if kind == "performance_degradation":
            events.append(
                Event(
                    kind=kind,
                    node_idx=int(rng.integers(0, topo.N)),
                    factor=float(rng.uniform(0.3, 0.7)),
                    time=t,
                )
            )
        elif kind == "service_failure":
            events.append(Event(kind=kind, node_idx=int(rng.integers(0, topo.N)), time=t))
        else:  # new_node_join
            events.append(
                Event(
                    kind=kind,
                    node_idx=next_join_idx,
                    capacity=float(rng.uniform(1500.0, 2500.0)),
                    ce=0.05,
                    distance=float(rng.uniform(*distance_range)),
                    time=t,
                )
            )
            next_join_idx += 1
    return events
