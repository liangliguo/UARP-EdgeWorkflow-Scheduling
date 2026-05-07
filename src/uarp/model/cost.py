"""Time / energy / deadline / success-rate formulas — paper §2.2.2–2.2.5.

Equations referenced inline. All vectorised over a single workflow.
"""

from __future__ import annotations

import numpy as np

from .edge import Topology
from .schedule import Schedule
from .workflow import Workflow


def transmission_time(wf: Workflow, topo: Topology, sched: Schedule, i: int) -> float:
    """OT(v_{m,i}) = b · d / BA (Eq. 4)."""
    k = sched.node_of(i)
    return wf.size(i) * topo.distances[k] / topo.BA


def execution_time(wf: Workflow, topo: Topology, sched: Schedule, i: int) -> float:
    """ET(v_{m,i}) = b / B_k (Eq. 5)."""
    k = sched.node_of(i)
    node = topo.node(k)
    if not node.available or node.capacity <= 0:
        return float("inf")
    return wf.size(i) / node.capacity


def schedule_times(wf: Workflow, topo: Topology, sched: Schedule) -> tuple[np.ndarray, np.ndarray]:
    """Compute ST and WT for every task (Eqs. 6–8) by topo order.

    Returns (ST, WT) arrays of length M.

    Reading of Eqs. 6–8: a task may begin only when all predecessors have
    finished AND its own data has been transmitted to the assigned node.
    For each predecessor p of i:
        ready_from_p = ST(p) + OT(p) + ET(p) + OT(i)
    ST(i) = max over predecessors of ready_from_p (0 if no predecessor).
    WT(i) = ST(i) + ET(i).
    """
    M = wf.M
    ST = np.zeros(M)
    WT = np.zeros(M)
    for i in wf.topo_order():
        preds = wf.predecessors(i)
        ot_i = transmission_time(wf, topo, sched, i)
        et_i = execution_time(wf, topo, sched, i)
        if not preds:
            ST[i] = ot_i
        else:
            ST[i] = max(WT[p] + ot_i for p in preds)
        WT[i] = ST[i] + et_i
    return ST, WT


def completion_time(wf: Workflow, topo: Topology, sched: Schedule) -> float:
    """WT(V_m) = WT of the last task in topo order."""
    _, WT = schedule_times(wf, topo, sched)
    return float(np.max(WT))


def transmission_energy(wf: Workflow, topo: Topology, sched: Schedule) -> float:
    """CMT (Eq. 9) — transmission energy.

    ct(v, z) is modelled as size · distance · ct_per_distance.
    """
    e = 0.0
    for i in range(wf.M):
        k = sched.node_of(i)
        e += wf.size(i) * topo.distances[k] * topo.ct_per_distance
    return e


def execution_energy(wf: Workflow, topo: Topology, sched: Schedule) -> float:
    """CME (Eq. 10) — execution energy = Σ b · ce(z)."""
    e = 0.0
    for i in range(wf.M):
        k = sched.node_of(i)
        e += wf.size(i) * topo.node(k).ce
    return e


def sync_energy(topo: Topology) -> float:
    """CMS (Eq. 11) — controller sync energy. Independent of schedule."""
    return float(np.sum(topo.h * topo.cs))


def total_energy(wf: Workflow, topo: Topology, sched: Schedule) -> float:
    """CM = CMT + CME + CMS (Eq. 12)."""
    return (
        transmission_energy(wf, topo, sched)
        + execution_energy(wf, topo, sched)
        + sync_energy(topo)
    )


def deadline(wf: Workflow, topo: Topology, sched: Schedule, alpha: float) -> float:
    """DT(V_m) = α · WT(v_{m,M}) (Eq. 13)."""
    return alpha * completion_time(wf, topo, sched)


def task_deadline(
    wf: Workflow,
    topo: Topology,
    sched: Schedule,
    i: int,
    critical_path: list[int],
) -> float:
    """DT(v_{m,i}) (Eq. 14) — linear interpolation on the critical path.

    DT(v_i) = (WT(v_q) - ST(v_p)) · (WT(v_i) - ST(v_p)) / (WT(v_i) - ST(v_p))
    where v_p, v_q are the first/last tasks on the critical path.

    NOTE: as written in the paper Eq. 14 is partially ambiguous; we follow
    the natural interpretation — proportionally allocate slack across tasks
    on the critical path.
    """
    ST, WT = schedule_times(wf, topo, sched)
    p, q = critical_path[0], critical_path[-1]
    span = WT[q] - ST[p]
    if span <= 0:
        return float(WT[i])
    return float((WT[i] - ST[p]) / span * (WT[q] - ST[p]) + ST[p])


def success_indicator(actual_finish: float, dt_value: float) -> int:
    """k(af) (Eq. 15) — 1 if finished before deadline."""
    return 1 if actual_finish < dt_value else 0


def success_rate(actual_finishes: np.ndarray, dt_values: np.ndarray) -> float:
    """SUC(V_m) = (1/T) Σ k(af) (Eq. 16) over T affected tasks."""
    T = len(actual_finishes)
    if T == 0:
        return 1.0
    return float(np.mean(actual_finishes < dt_values))
