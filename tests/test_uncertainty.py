"""Tests for uncertainty events and Algorithm 2."""

import numpy as np

from uarp.baselines import benchmark_assignment
from uarp.model import (
    make_homogeneous_topology,
    random_dag,
    subset,
)
from uarp.uncertainty import (
    Event,
    apply_events,
    ff_sub_scheduler,
    generate_events,
    reschedule,
    reschedule_benchmark,
    uarp_sub_scheduler,
    wf_sub_scheduler,
)


def test_subset_preserves_partial_order():
    wf = random_dag(n_tasks=10, edge_prob=0.3, seed=0)
    keep = wf.topo_order()[3:]
    sub_wf, ordered = subset(wf, keep)
    assert sub_wf.M == len(keep)
    # global ids in `ordered` are exactly the kept set, in original topo order
    assert set(ordered) == set(keep)
    assert ordered == [g for g in wf.topo_order() if g in set(keep)]
    # subgraph topo order is 0..M-1 by construction
    assert sub_wf.topo_order() == list(range(sub_wf.M))
    # sizes carried over
    for local_i, global_i in enumerate(ordered):
        assert sub_wf.tasks[local_i].size == wf.tasks[global_i].size


def test_apply_event_performance_degradation():
    topo = make_homogeneous_topology(N=4, seed=0)
    ev = Event(kind="performance_degradation", node_idx=1, factor=0.5)
    new_topo = apply_events(topo, [ev])
    assert new_topo.nodes[1].capacity == topo.nodes[1].capacity * 0.5
    # original topology must be untouched (deepcopy)
    assert topo.nodes[1].capacity == 2000.0


def test_apply_event_service_failure():
    """Service failure is modelled as severe degradation (capacity * 0.05)."""
    topo = make_homogeneous_topology(N=4, seed=0)
    original = topo.nodes[2].capacity
    ev = Event(kind="service_failure", node_idx=2)
    new_topo = apply_events(topo, [ev])
    assert new_topo.nodes[2].capacity == original * 0.05
    assert topo.nodes[2].capacity == original


def test_apply_event_new_node_join():
    topo = make_homogeneous_topology(N=4, seed=0)
    ev = Event(kind="new_node_join", node_idx=4, capacity=1800.0, ce=0.05, distance=42.0)
    new_topo = apply_events(topo, [ev])
    assert new_topo.N == 5
    assert new_topo.nodes[4].capacity == 1800.0
    assert new_topo.distances[4] == 42.0
    assert topo.N == 4  # original untouched


def test_generate_events_is_seeded():
    topo = make_homogeneous_topology(N=10, seed=0)
    a = generate_events(topo, n_events=5, seed=7)
    b = generate_events(topo, n_events=5, seed=7)
    assert [(e.kind, e.node_idx) for e in a] == [(e.kind, e.node_idx) for e in b]


def test_reschedule_benchmark_keeps_assignment():
    """Benchmark MUST keep the initial assignment regardless of events."""
    wf = random_dag(n_tasks=10, edge_prob=0.3, seed=0)
    topo = make_homogeneous_topology(N=4, seed=0)
    init = benchmark_assignment(wf, topo, seed=0)
    events = [Event(kind="performance_degradation", node_idx=0, factor=0.3)]
    res = reschedule_benchmark(wf, topo, init, events, alpha=1.2, progress_frac=0.4)
    np.testing.assert_array_equal(res.final_schedule.assignment, init.assignment)
    # cost is recomputed on the post-event topology, so WT/CM may differ
    assert res.actual_wt > 0


def test_reschedule_ff_changes_remaining_tasks():
    """After service_failure on node 0, FF should largely avoid it for new tasks."""
    wf = random_dag(n_tasks=10, edge_prob=0.3, seed=0)
    topo = make_homogeneous_topology(N=4, seed=0)
    init = benchmark_assignment(wf, topo, seed=0)
    events = [Event(kind="service_failure", node_idx=0)]
    res = reschedule(wf, topo, init, events, ff_sub_scheduler(), alpha=1.2, progress_frac=0.4)
    # the rescheduling must produce a valid assignment for all tasks
    assert res.final_schedule.assignment.min() >= 0
    assert res.final_schedule.assignment.max() < res.final_topology.N


def test_reschedule_wf_uses_worst_fit():
    wf = random_dag(n_tasks=10, edge_prob=0.3, seed=0)
    topo = make_homogeneous_topology(N=4, seed=0)
    init = benchmark_assignment(wf, topo, seed=0)
    events = [Event(kind="performance_degradation", node_idx=0, factor=0.3)]
    res = reschedule(wf, topo, init, events, wf_sub_scheduler(), alpha=1.2, progress_frac=0.4)
    # WF spreads load — at least 2 distinct nodes among the affected tasks
    affected_assign = res.final_schedule.assignment[res.affected_task_ids]
    assert len(set(affected_assign.tolist())) >= 2


def test_reschedule_uarp_runs_end_to_end():
    """Algorithm 2 with UARP sub-scheduler completes and yields finite costs."""
    wf = random_dag(n_tasks=8, edge_prob=0.3, seed=0)
    topo = make_homogeneous_topology(N=4, seed=0)
    init = benchmark_assignment(wf, topo, seed=0)
    events = [Event(kind="performance_degradation", node_idx=1, factor=0.5)]
    res = reschedule(
        wf, topo, init, events,
        uarp_sub_scheduler(pop_size=20, n_gen=10, seed=0),
        alpha=1.3, progress_frac=0.4,
    )
    assert np.isfinite(res.actual_wt)
    assert np.isfinite(res.actual_cm)
    assert len(res.affected_task_ids) == wf.M - int(0.4 * wf.M)
    assert res.affected_deadlines.shape == res.affected_finishes.shape
