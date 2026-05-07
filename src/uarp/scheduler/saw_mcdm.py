"""Simple Additive Weighting (SAW) + MCDM — paper §3.2 Eqs. (19-23).

Given a Pareto front F of shape (P, 2) with (WT, CM):
    WT^t = WT - WT_min                   (Eq. 19)
    CM^t = CM - CM_min                   (Eq. 20)
    WT^n = WT^t / σ_wt                   (Eq. 21)
    CM^n = CM^t / σ_cm                   (Eq. 22)
    W    = λ_wt · WT^n + λ_cm · CM^n     (Eq. 23)
The optimum is the solution that minimises W (smallest normalised cost).

NOTE: paper writes "highest utility value" but its W is a cost — minimum is
best. We expose `utility(...)` as 1 − W so larger means better, matching
Figure 5's bar-chart orientation.
"""

from __future__ import annotations

import numpy as np


def normalise(F: np.ndarray) -> np.ndarray:
    """Linear min-extreme normalisation (Eqs. 19-22). Returns array of same shape."""
    F = np.asarray(F, dtype=float)
    F_t = F - F.min(axis=0, keepdims=True)
    sigma = np.where(F_t.max(axis=0) > 0, F_t.max(axis=0), 1.0)
    return F_t / sigma


def saw_cost(F: np.ndarray, weights: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """W = λ_wt · WT^n + λ_cm · CM^n (Eq. 23). Lower is better."""
    F_n = normalise(F)
    w = np.asarray(weights, dtype=float)
    return F_n @ w


def utility(F: np.ndarray, weights: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """Larger = better, in [0, 1]. Used for Figure 5 bars."""
    return 1.0 - saw_cost(F, weights)


def best_index(F: np.ndarray, weights: tuple[float, float] = (0.5, 0.5)) -> int:
    """Index of the most balanced solution (max utility)."""
    return int(np.argmax(utility(F, weights)))
