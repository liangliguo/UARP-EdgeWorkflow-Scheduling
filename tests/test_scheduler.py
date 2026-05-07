"""Smoke tests for NSGA-III + SAW/MCDM pipeline."""

import numpy as np

from uarp.model import Schedule, completion_time, make_homogeneous_topology, random_dag, total_energy
from uarp.scheduler import best_index, normalise, saw_cost, solve, utility


def test_normalise_zero_when_constant():
    F = np.array([[1.0, 2.0], [1.0, 2.0]])
    out = normalise(F)
    np.testing.assert_allclose(out, np.zeros_like(F))


def test_normalise_in_unit_box():
    F = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    out = normalise(F)
    assert out.min() == 0.0
    assert out.max() == 1.0


def test_saw_picks_balanced_solution():
    F = np.array([[10.0, 1.0], [5.0, 5.0], [1.0, 10.0]])
    # under equal weights, the middle is balanced
    assert best_index(F, (0.5, 0.5)) == 1
    util = utility(F, (0.5, 0.5))
    assert util[1] >= util[0] and util[1] >= util[2]


def test_saw_cost_lower_for_better():
    F = np.array([[1.0, 1.0], [10.0, 10.0]])
    cost = saw_cost(F, (0.5, 0.5))
    assert cost[0] < cost[1]


def test_solve_smoke_small_workflow():
    """Run NSGA-III on a tiny problem and assert output shapes + feasibility."""
    wf = random_dag(n_tasks=8, edge_prob=0.3, seed=42)
    topo = make_homogeneous_topology(N=4, seed=42)
    res = solve(wf, topo, pop_size=20, n_gen=10, n_partitions=6, seed=0)
    assert res.pareto_F.shape[1] == 2
    assert res.pareto_X.shape == (res.pareto_F.shape[0], wf.M)
    # the chosen schedule must agree with the recomputed cost
    wt = completion_time(wf, topo, res.best)
    cm = total_energy(wf, topo, res.best)
    np.testing.assert_allclose(res.best_F, [wt, cm], rtol=1e-9)
    # all chromosomes should be valid node indices
    assert res.pareto_X.min() >= 0
    assert res.pareto_X.max() < topo.N


def test_solve_with_deadline_constraint():
    """Pass a generous deadline; all returned solutions should satisfy it."""
    wf = random_dag(n_tasks=8, edge_prob=0.3, seed=7)
    topo = make_homogeneous_topology(N=4, seed=7)
    # establish a baseline WT first, then set a very loose deadline
    base = solve(wf, topo, pop_size=20, n_gen=10, n_partitions=6, seed=0)
    loose_dt = float(base.pareto_F[:, 0].max() * 2.0)
    res = solve(wf, topo, deadline=loose_dt, pop_size=20, n_gen=10, n_partitions=6, seed=0)
    assert (res.pareto_F[:, 0] <= loose_dt + 1e-6).all()
