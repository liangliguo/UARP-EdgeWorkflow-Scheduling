"""Algorithm 2 — dynamic resource provisioning under uncertainty (paper §3.4).

Workflow execution is simulated as: a prefix of tasks (in topological order)
finishes before any uncertainty fires. When events trigger, the remaining tasks
form a new sub-workflow that is rescheduled on the post-event topology.

The same skeleton is reused by FF/WF for fairness — only the inner scheduler
differs. Benchmark = no rescheduling at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from uarp.baselines import first_fit_assignment, worst_fit_assignment
from uarp.model import (
    Schedule,
    Topology,
    Workflow,
    completion_time,
    subset,
    total_energy,
)
from uarp.scheduler import solve as uarp_solve

from .events import Event, apply_events


SubScheduler = Callable[[Workflow, Topology], Schedule]


@dataclass
class RescheduleResult:
    final_schedule: Schedule
    final_topology: Topology
    actual_wt: float
    actual_cm: float
    affected_task_ids: list[int]
    affected_finishes: np.ndarray
    affected_deadlines: np.ndarray


def _split_progress(workflow: Workflow, progress_frac: float) -> tuple[list[int], list[int]]:
    """Return (completed_global_ids, remaining_global_ids) under topo order."""
    order = workflow.topo_order()
    n_done = int(round(progress_frac * len(order)))
    return order[:n_done], order[n_done:]


def _per_task_deadlines(
    workflow: Workflow,
    topology: Topology,
    reference_schedule: Schedule,
    alpha: float,
    affected_ids: list[int],
) -> np.ndarray:
    """Per-task deadlines for affected tasks (Eqs. 13-14).

    The reference schedule should be the *ideal* (uncertainty-free) plan and is
    shared across all comparison methods so they are evaluated against the
    same DT — matching paper §2.2.4 where α scales the ideal completion time.
    """
    from uarp.model.cost import schedule_times

    _, WT_ideal = schedule_times(workflow, topology, reference_schedule)
    return alpha * WT_ideal[affected_ids]


def reschedule(
    workflow: Workflow,
    topology: Topology,
    initial_schedule: Schedule,
    events: list[Event],
    sub_scheduler: SubScheduler,
    *,
    alpha: float = 1.2,
    progress_frac: float = 0.4,
    deadline_reference: Schedule | None = None,
) -> RescheduleResult:
    """Run Algorithm 2's flow with a pluggable sub-scheduler for the remainder.

    The initial_schedule's assignment is kept for completed tasks; remaining
    tasks are reassigned by `sub_scheduler` on the post-event topology.
    """
    completed_ids, remaining_ids = _split_progress(workflow, progress_frac)
    new_topo = apply_events(topology, events)
    if remaining_ids:
        sub_wf, ordered_ids = subset(workflow, remaining_ids)
        sub_assignment = sub_scheduler(sub_wf, new_topo)
    else:
        ordered_ids, sub_assignment = [], Schedule(assignment=np.zeros(0, dtype=int))

    final_assignment = initial_schedule.assignment.copy()
    # Map back to global indices.
    for local_i, global_i in enumerate(ordered_ids):
        final_assignment[global_i] = sub_assignment.assignment[local_i]
    # Repair: any "completed" task that was originally assigned to a now-
    # unavailable node must still be considered as having executed there
    # (its OT/ET have already been spent), so we leave its index alone.
    final_schedule = Schedule(assignment=final_assignment)

    actual_wt = completion_time(workflow, new_topo, final_schedule)
    actual_cm = total_energy(workflow, new_topo, final_schedule)

    affected_finishes = np.array(
        [
            _task_finish_time(workflow, new_topo, final_schedule, i)
            for i in remaining_ids
        ]
    )
    ref = deadline_reference if deadline_reference is not None else initial_schedule
    affected_deadlines = _per_task_deadlines(
        workflow, topology, ref, alpha, remaining_ids
    )
    return RescheduleResult(
        final_schedule=final_schedule,
        final_topology=new_topo,
        actual_wt=actual_wt,
        actual_cm=actual_cm,
        affected_task_ids=remaining_ids,
        affected_finishes=affected_finishes,
        affected_deadlines=affected_deadlines,
    )


def _task_finish_time(
    workflow: Workflow, topology: Topology, sched: Schedule, i: int
) -> float:
    from uarp.model.cost import schedule_times

    _, WT = schedule_times(workflow, topology, sched)
    return float(WT[i])


# ---------------- helpers binding the sub-scheduler ---------------- #


def uarp_sub_scheduler(
    *,
    pop_size: int = 50,
    n_gen: int = 40,
    seed: int = 0,
) -> SubScheduler:
    def _run(wf: Workflow, topo: Topology) -> Schedule:
        return uarp_solve(wf, topo, pop_size=pop_size, n_gen=n_gen, seed=seed).best

    return _run


def ff_sub_scheduler() -> SubScheduler:
    return lambda wf, topo: first_fit_assignment(wf, topo)


def wf_sub_scheduler() -> SubScheduler:
    return lambda wf, topo: worst_fit_assignment(wf, topo)


def benchmark_sub_scheduler() -> SubScheduler:
    """Benchmark does NOT reschedule — keep whatever the initial strategy was.

    The skeleton still calls the sub-scheduler on the remaining sub-DAG. To
    represent "no rescheduling" we map every remaining task to the placeholder
    node 0; the caller then explicitly bypasses the subroutine and keeps the
    original initial_schedule. See `reschedule_benchmark` below.
    """
    raise RuntimeError("Benchmark does not reschedule; use reschedule_benchmark()")


def reschedule_benchmark(
    workflow: Workflow,
    topology: Topology,
    initial_schedule: Schedule,
    events: list[Event],
    *,
    alpha: float = 1.2,
    progress_frac: float = 0.4,
    deadline_reference: Schedule | None = None,
) -> RescheduleResult:
    """Benchmark — keep initial_schedule unchanged, just measure post-event cost.

    If a service_failure event renders a task's assigned node unavailable, the
    task is transparently redirected to the first available node. This is the
    minimum repair needed to keep the workflow executable; Benchmark still
    does NOT optimise the assignment of remaining tasks.
    """
    _, remaining_ids = _split_progress(workflow, progress_frac)
    new_topo = apply_events(topology, events)
    repaired = initial_schedule.assignment.copy()
    available = [k for k in range(new_topo.N) if new_topo.node(k).available]
    if available:
        fallback = available[0]
        for i in range(workflow.M):
            if not new_topo.node(repaired[i]).available:
                repaired[i] = fallback
    repaired_sched = Schedule(assignment=repaired)
    actual_wt = completion_time(workflow, new_topo, repaired_sched)
    actual_cm = total_energy(workflow, new_topo, repaired_sched)
    affected_finishes = np.array(
        [
            _task_finish_time(workflow, new_topo, repaired_sched, i)
            for i in remaining_ids
        ]
    )
    ref = deadline_reference if deadline_reference is not None else initial_schedule
    affected_deadlines = _per_task_deadlines(
        workflow, topology, ref, alpha, remaining_ids
    )
    return RescheduleResult(
        final_schedule=repaired_sched,
        final_topology=new_topo,
        actual_wt=actual_wt,
        actual_cm=actual_cm,
        affected_task_ids=remaining_ids,
        affected_finishes=affected_finishes,
        affected_deadlines=affected_deadlines,
    )
