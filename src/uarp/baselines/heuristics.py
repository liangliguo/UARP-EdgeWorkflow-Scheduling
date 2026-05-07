"""Three baselines from paper §4.1.

- Benchmark: fixed initial assignment, no rescheduling on uncertainty.
- FF (First Fit): for each task, pick the first node that can serve it.
- WF (Worst Fit): for each task, pick the node with the most spare capacity.

The paper says the FF/WF heuristics regenerate strategies when uncertainty
arises; the rescheduling logic is shared with UARP via uncertainty/algorithm2.
This module only produces *initial* assignments.
"""

from __future__ import annotations

import numpy as np

from uarp.model import Schedule, Topology, Workflow


def benchmark_assignment(
    workflow: Workflow,
    topology: Topology,
    *,
    seed: int = 0,
) -> Schedule:
    """Random fixed assignment — Benchmark per §4.1(1).

    The paper does not specify how Benchmark picks its initial mapping; we use
    a seeded random uniform pick so the comparison is reproducible.
    """
    rng = np.random.default_rng(seed)
    return Schedule(assignment=rng.integers(0, topology.N, size=workflow.M))


def first_fit_assignment(
    workflow: Workflow,
    topology: Topology,
) -> Schedule:
    """FF — assign each task to the first node that has spare capacity.

    Capacity is tracked as cumulative size already mapped to each node; once a
    node's load exceeds its processing capability it is considered "saturated"
    and the next available node is taken. If all are saturated, wrap around.
    """
    N = topology.N
    load = np.zeros(N)
    assignment = np.zeros(workflow.M, dtype=int)
    for i in workflow.topo_order():
        size = workflow.size(i)
        chosen = -1
        for k in range(N):
            cap = topology.node(k).capacity
            if topology.node(k).available and load[k] + size <= cap:
                chosen = k
                break
        if chosen < 0:
            # all saturated — fall back to least-loaded available node
            avail = [k for k in range(N) if topology.node(k).available]
            chosen = int(avail[np.argmin(load[avail])]) if avail else 0
        assignment[i] = chosen
        load[chosen] += size
    return Schedule(assignment=assignment)


def worst_fit_assignment(
    workflow: Workflow,
    topology: Topology,
) -> Schedule:
    """WF — assign each task to the node with the most remaining capacity."""
    N = topology.N
    load = np.zeros(N)
    assignment = np.zeros(workflow.M, dtype=int)
    for i in workflow.topo_order():
        spare = np.array(
            [
                (topology.node(k).capacity - load[k]) if topology.node(k).available else -np.inf
                for k in range(N)
            ]
        )
        chosen = int(np.argmax(spare))
        assignment[i] = chosen
        load[chosen] += workflow.size(i)
    return Schedule(assignment=assignment)
