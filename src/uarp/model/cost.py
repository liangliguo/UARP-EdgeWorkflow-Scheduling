"""Time / energy / deadline / success-rate formulas — paper §2.2.2–2.2.5.

Equations referenced inline. All vectorised over a single workflow.
"""

from __future__ import annotations

import numpy as np

from .edge import Topology
from .schedule import Schedule
from .workflow import Workflow


def transmission_time(
    wf: Workflow,
    topo: Topology,
    sched: Schedule,
    i: int,
    t_start: float = 0.0,
) -> float:
    """OT(v_{m,i}) = b · d(t_start) / BA (Eq. 4 with 一.1 mobility).

    When ``topo.mobility`` is set, distance is evaluated at ``t_start`` — the
    moment the data starts being transmitted. With no mobility this reduces
    to the constant-distance formula.
    """
    k = sched.node_of(i)
    return wf.size(i) * topo.distance_at(t_start, k) / topo.BA


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
        # 一.1: data starts being transmitted once predecessors finish; use
        # that moment as t_start so a mobility-aware topology samples the
        # right distance.
        pred_ready = max((WT[p] for p in preds), default=0.0)
        ot_i = transmission_time(wf, topo, sched, i, t_start=pred_ready)
        et_i = execution_time(wf, topo, sched, i)
        ST[i] = pred_ready + ot_i if preds else ot_i
        WT[i] = ST[i] + et_i
    return ST, WT


def completion_time(wf: Workflow, topo: Topology, sched: Schedule) -> float:
    """WT(V_m) = WT of the last task in topo order."""
    _, WT = schedule_times(wf, topo, sched)
    return float(np.max(WT))


def transmission_energy(wf: Workflow, topo: Topology, sched: Schedule) -> float:
    """CMT (Eq. 9) — transmission energy.

    ct(v, z) is modelled as size · distance · ct_per_distance. With
    mobility (一.1), distance is evaluated at the task's predecessor-ready
    time so the cost matches what the schedule actually pays in OT.
    """
    if topo.mobility is None:
        e = 0.0
        for i in range(wf.M):
            k = sched.node_of(i)
            e += wf.size(i) * topo.distances[k] * topo.ct_per_distance
        return e
    _, WT = schedule_times(wf, topo, sched)
    e = 0.0
    for i in range(wf.M):
        k = sched.node_of(i)
        # OT(i) starts once all predecessors have finished executing.
        pred_ready = max((float(WT[p]) for p in wf.predecessors(i)), default=0.0)
        e += wf.size(i) * topo.distance_at(pred_ready, k) * topo.ct_per_distance
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


def task_deadline_legacy(
    wf: Workflow,
    topo: Topology,
    sched: Schedule,
    i: int,
    critical_path: list[int],
) -> float:
    """DT(v_{m,i}) (Eq. 14, paper-as-written) — ill-posed; kept for regression.

    Paper writes:
      DT(v_i) = (WT(v_q) - ST(v_p)) · (WT(v_i) - ST(v_p)) / (WT(v_i) - ST(v_p))
    where v_p, v_q are the first/last tasks on the critical path. Numerator
    and denominator share the (WT(v_i) - ST(v_p)) factor, so the formula
    collapses. See IMPROVEMENTS.md #二.7 for the diagnosis.
    """
    ST, WT = schedule_times(wf, topo, sched)
    p, q = critical_path[0], critical_path[-1]
    span = WT[q] - ST[p]
    if span <= 0:
        return float(WT[i])
    return float((WT[i] - ST[p]) / span * (WT[q] - ST[p]) + ST[p])


# Back-compat alias — historical callers used this name.
task_deadline = task_deadline_legacy


def task_deadlines_slf(
    wf: Workflow,
    topo: Topology,
    sched: Schedule,
    alpha: float,
) -> np.ndarray:
    """Per-task deadlines via Slack-Time Allocation (Hwang et al. 1989).

    Replaces the ill-posed paper Eq. 14. Allocates the total slack
    ``(α - 1) · WT_max`` across all tasks in proportion to each task's
    execution duration ``ET[i] = WT[i] - ST[i]``.

    Returns an array of length M with DT[i] = WT_ideal[i] + slack_i.
    """
    ST, WT = schedule_times(wf, topo, sched)
    WT_max = float(WT.max()) if len(WT) else 0.0
    total_slack = (float(alpha) - 1.0) * WT_max
    ET = WT - ST
    et_total = float(ET.sum())
    if et_total <= 0.0:
        return WT.copy()
    return WT + total_slack * (ET / et_total)


def success_indicator(actual_finish: float, dt_value: float) -> int:
    """k(af) (Eq. 15) — 1 if finished before deadline."""
    return 1 if actual_finish < dt_value else 0


def success_rate(actual_finishes: np.ndarray, dt_values: np.ndarray) -> float:
    """SUC(V_m) = (1/T) Σ k(af) (Eq. 16) over T affected tasks."""
    T = len(actual_finishes)
    if T == 0:
        return 1.0
    return float(np.mean(actual_finishes < dt_values))
