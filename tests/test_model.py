"""Unit tests for paper formulas (Eqs. 4-12)."""

import networkx as nx
import numpy as np

from uarp.model import (
    EdgeNode,
    Schedule,
    Task,
    Topology,
    Workflow,
    completion_time,
    execution_energy,
    execution_time,
    schedule_times,
    success_rate,
    sync_energy,
    total_energy,
    transmission_energy,
    transmission_time,
)


def _toy_topology(n_nodes: int = 2) -> Topology:
    nodes = [EdgeNode(idx=k, capacity=100.0, ce=0.1) for k in range(n_nodes)]
    distances = np.array([10.0, 20.0][:n_nodes])
    h = np.array([[0, 1], [1, 0]])
    cs = np.array([[0.0, 0.5], [0.5, 0.0]])
    return Topology(
        nodes=nodes,
        BA=1000.0,
        distances=distances,
        ct_per_distance=0.001,
        h=h,
        cs=cs,
    )


def _serial_workflow(sizes: list[float]) -> Workflow:
    g = nx.DiGraph()
    g.add_nodes_from(range(len(sizes)))
    for i in range(len(sizes) - 1):
        g.add_edge(i, i + 1)
    return Workflow(tasks=[Task(idx=i, size=s) for i, s in enumerate(sizes)], graph=g)


def _parallel_workflow(sizes: list[float]) -> Workflow:
    """Source -> {middle tasks} -> sink."""
    n = len(sizes)
    assert n >= 3
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(1, n - 1):
        g.add_edge(0, i)
        g.add_edge(i, n - 1)
    return Workflow(tasks=[Task(idx=i, size=s) for i, s in enumerate(sizes)], graph=g)


def test_eq4_transmission_time():
    """OT = b · d / BA. b=200, d=10, BA=1000 -> 2.0"""
    wf = _serial_workflow([200.0])
    topo = _toy_topology(1)
    sched = Schedule.from_list([0])
    assert transmission_time(wf, topo, sched, 0) == 200.0 * 10.0 / 1000.0


def test_eq5_execution_time():
    """ET = b / B_k. b=200, B=100 -> 2.0"""
    wf = _serial_workflow([200.0])
    topo = _toy_topology(1)
    sched = Schedule.from_list([0])
    assert execution_time(wf, topo, sched, 0) == 200.0 / 100.0


def test_serial_schedule_times():
    """3 tasks in series, all on node 0. b=[100,100,100], B=100, d=10, BA=1000.
    OT each = 1.0, ET each = 1.0.
    ST: t0=1, t1=ST(p)+OT(i) where p=t0 finishes at WT(t0)=2. t1 starts at 2+1=3, WT=4.
    t2 starts at 4+1=5, WT=6.
    """
    wf = _serial_workflow([100.0, 100.0, 100.0])
    topo = _toy_topology(1)
    sched = Schedule.from_list([0, 0, 0])
    ST, WT = schedule_times(wf, topo, sched)
    np.testing.assert_allclose(ST, [1.0, 3.0, 5.0])
    np.testing.assert_allclose(WT, [2.0, 4.0, 6.0])
    assert completion_time(wf, topo, sched) == 6.0


def test_parallel_schedule_times():
    """Source + 2 middle + sink. Middle tasks parallel; sink waits for slowest.
    sizes=[100,100,200,100], B=100, d=10, BA=1000.
    OT (per task) = size·10/1000 -> [1, 1, 2, 1]; ET = size/100 -> [1, 1, 2, 1].
    t0: ST=OT(0)=1, WT=2.
    t1: ST=WT(t0)+OT(1)=2+1=3, WT=4.
    t2: ST=WT(t0)+OT(2)=2+2=4, WT=6.
    t3: ST=max(WT(t1)+OT(3), WT(t2)+OT(3))=max(5,7)=7, WT=8.
    """
    wf = _parallel_workflow([100.0, 100.0, 200.0, 100.0])
    topo = _toy_topology(1)
    sched = Schedule.from_list([0, 0, 0, 0])
    ST, WT = schedule_times(wf, topo, sched)
    assert WT[0] == 2.0
    assert WT[1] == 4.0
    assert WT[2] == 6.0
    assert WT[3] == 8.0


def test_eq9_transmission_energy():
    wf = _serial_workflow([100.0, 200.0])
    topo = _toy_topology(2)
    sched = Schedule.from_list([0, 1])
    # CMT = 100·10·0.001 + 200·20·0.001 = 1.0 + 4.0 = 5.0
    assert transmission_energy(wf, topo, sched) == 5.0


def test_eq10_execution_energy():
    wf = _serial_workflow([100.0, 200.0])
    topo = _toy_topology(2)
    sched = Schedule.from_list([0, 1])
    # CME = 100·0.1 + 200·0.1 = 30.0
    assert execution_energy(wf, topo, sched) == 30.0


def test_eq11_sync_energy():
    topo = _toy_topology(2)
    # h = [[0,1],[1,0]], cs = [[0,0.5],[0.5,0]] -> 0.5+0.5 = 1.0
    assert sync_energy(topo) == 1.0


def test_eq12_total_energy():
    wf = _serial_workflow([100.0, 200.0])
    topo = _toy_topology(2)
    sched = Schedule.from_list([0, 1])
    assert total_energy(wf, topo, sched) == 5.0 + 30.0 + 1.0


def test_eq16_success_rate():
    af = np.array([1.0, 2.0, 5.0, 8.0])
    dt = np.array([2.0, 1.5, 6.0, 7.0])
    # 1<2 ✓, 2<1.5 ✗, 5<6 ✓, 8<7 ✗ -> 2/4 = 0.5
    assert success_rate(af, dt) == 0.5
