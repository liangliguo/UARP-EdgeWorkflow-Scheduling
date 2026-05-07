"""Tests for FF/WF/Benchmark baselines."""

import numpy as np

from uarp.baselines import benchmark_assignment, first_fit_assignment, worst_fit_assignment
from uarp.model import EdgeNode, Topology, Workflow, completion_time, random_dag


def _toy_topo(capacities: list[float]) -> Topology:
    nodes = [EdgeNode(idx=k, capacity=c, ce=0.05) for k, c in enumerate(capacities)]
    distances = np.full(len(capacities), 10.0)
    return Topology(
        nodes=nodes,
        BA=1000.0,
        distances=distances,
        ct_per_distance=1e-5,
        h=np.zeros((1, 1), dtype=int),
        cs=np.zeros((1, 1)),
    )


def test_first_fit_fills_first_node_first():
    """With small tasks and abundant capacity, FF keeps using node 0."""
    topo = _toy_topo([10000.0, 10000.0])
    wf = random_dag(n_tasks=5, edge_prob=0.2, seed=0)
    sched = first_fit_assignment(wf, topo)
    assert (sched.assignment == 0).all()


def test_first_fit_overflows_to_next_node():
    """When node 0 saturates, FF moves to node 1."""
    # tasks ~50-200 in size; tiny node-0 capacity forces overflow
    topo = _toy_topo([100.0, 10000.0])
    wf = random_dag(n_tasks=5, edge_prob=0.2, seed=0)
    sched = first_fit_assignment(wf, topo)
    # at least one task should land on node 1
    assert (sched.assignment == 1).any()


def test_worst_fit_distributes_across_nodes():
    """WF should distribute across all 4 nodes for a 10-task workflow."""
    topo = _toy_topo([10000.0] * 4)
    wf = random_dag(n_tasks=10, edge_prob=0.2, seed=0)
    sched = worst_fit_assignment(wf, topo)
    assert len(np.unique(sched.assignment)) >= 3


def test_benchmark_is_seeded():
    topo = _toy_topo([10000.0] * 4)
    wf = random_dag(n_tasks=8, edge_prob=0.2, seed=0)
    a = benchmark_assignment(wf, topo, seed=42)
    b = benchmark_assignment(wf, topo, seed=42)
    np.testing.assert_array_equal(a.assignment, b.assignment)


def test_baselines_produce_valid_indices():
    topo = _toy_topo([10000.0] * 4)
    wf = random_dag(n_tasks=10, edge_prob=0.3, seed=0)
    for fn in (
        lambda: benchmark_assignment(wf, topo, seed=0),
        lambda: first_fit_assignment(wf, topo),
        lambda: worst_fit_assignment(wf, topo),
    ):
        s = fn()
        assert s.assignment.min() >= 0
        assert s.assignment.max() < topo.N
        # the schedule must be evaluable
        wt = completion_time(wf, topo, s)
        assert wt > 0
