"""Multi-objective workflow scheduling problem for pymoo (paper §3.1).

Chromosome: integer vector of length M, gene[i] ∈ [0, N-1] = node assignment.
Fitness 1: WT(v_M)  — completion time (Eq. 8)
Fitness 2: CM(V_m)  — total energy (Eq. 12)
Constraint: WT(V_m) ≤ DT(V_m)  (Eq. 18)
"""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem

from uarp.model import (
    Schedule,
    Topology,
    Workflow,
    completion_time,
    total_energy,
)


class WorkflowSchedulingProblem(Problem):
    """Two-objective integer-encoded scheduling problem.

    Parameters
    ----------
    workflow : Workflow
    topology : Topology
    deadline : float | None
        If given, used as DT(V_m) for the constraint WT ≤ DT.
        If None, no constraint is applied (open optimisation).
    """

    def __init__(
        self,
        workflow: Workflow,
        topology: Topology,
        deadline: float | None = None,
    ):
        self.workflow = workflow
        self.topology = topology
        self.deadline = deadline
        super().__init__(
            n_var=workflow.M,
            n_obj=2,
            n_ieq_constr=0 if deadline is None else 1,
            xl=0,
            xu=topology.N - 1,
            vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        F = np.zeros((n_pop, 2))
        G = np.zeros((n_pop, 1)) if self.deadline is not None else None
        for p in range(n_pop):
            sched = Schedule(assignment=np.asarray(X[p], dtype=int))
            wt = completion_time(self.workflow, self.topology, sched)
            cm = total_energy(self.workflow, self.topology, sched)
            F[p, 0] = wt
            F[p, 1] = cm
            if G is not None:
                G[p, 0] = wt - self.deadline  # ≤ 0 means feasible
        out["F"] = F
        if G is not None:
            out["G"] = G
