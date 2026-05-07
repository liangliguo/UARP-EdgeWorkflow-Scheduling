"""UARP scheduler: NSGA-III over the workflow problem + SAW/MCDM (paper §3, Algorithm 1).

This is the main entry point. Algorithm 2 (uncertainty-aware rescheduling) lives
in src/uarp/uncertainty/algorithm2.py and reuses `solve()` from this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from uarp.model import Schedule, Topology, Workflow

from .problem import WorkflowSchedulingProblem
from .saw_mcdm import best_index, utility


@dataclass
class UARPResult:
    """Output of one Algorithm 1 invocation."""

    pareto_F: np.ndarray  # shape (P, 2) — (WT, CM) of non-dominated solutions
    pareto_X: np.ndarray  # shape (P, M) — chromosomes
    best: Schedule  # most balanced strategy via SAW/MCDM
    best_F: np.ndarray  # shape (2,) — (WT, CM) of `best`
    utilities: np.ndarray  # shape (P,) — utility per Pareto solution


def solve(
    workflow: Workflow,
    topology: Topology,
    *,
    deadline: float | None = None,
    pop_size: int = 100,
    n_gen: int = 100,
    n_partitions: int = 12,
    crossover_prob: float = 0.9,
    mutation_eta: int = 20,
    seed: int = 0,
    weights: tuple[float, float] = (0.5, 0.5),
) -> UARPResult:
    """Run NSGA-III + SAW/MCDM and return the chosen strategy + Pareto front.

    Defaults are documented in TODO §4 (paper does not pin them).
    """
    problem = WorkflowSchedulingProblem(workflow, topology, deadline=deadline)
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=n_partitions)
    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=crossover_prob, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0 / max(workflow.M, 1), eta=mutation_eta, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    F = np.atleast_2d(res.F)
    X = np.atleast_2d(res.X).astype(int)
    util = utility(F, weights)
    bi = best_index(F, weights)
    best_sched = Schedule(assignment=X[bi].copy())
    return UARPResult(
        pareto_F=F,
        pareto_X=X,
        best=best_sched,
        best_F=F[bi].copy(),
        utilities=util,
    )
