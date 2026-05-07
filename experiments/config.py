"""Shared experiment configuration — paper Table 2 + reproduction defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    # Paper Table 2
    BA: float = 1000.0  # mbps
    vm_capacity: float = 2000.0  # MHz
    n_edge_nodes: int = 20
    theta_start: float = 0.2  # kW (unused in time/energy formulas, kept for ref)
    theta_idle: float = 0.03
    theta_op: float = 0.05  # used as ce(z_k) — operating power per unit data
    # Workflow scales
    task_scales: tuple[int, ...] = (10, 15, 20, 25, 30, 35)
    # NSGA-III hyper-params (chosen by us — paper does not specify)
    pop_size: int = 100
    n_gen: int = 100
    n_partitions: int = 12
    # Algorithm 2
    progress_frac: float = 0.4
    n_uncertainty_events: int = 1
    # Statistical
    n_repeats: int = 5  # repeats per (scale, method) — keep small for fast runs
    # Deadline coefficients for Figure 9
    alpha_grid: tuple[float, ...] = (1.1, 1.2, 1.3, 1.4)
    # DAG generation
    edge_prob: float = 0.3
    size_range: tuple[float, float] = (50.0, 200.0)
    # Reproducibility
    seed: int = 2020


CFG = ExperimentConfig()
