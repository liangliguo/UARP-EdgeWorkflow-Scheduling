"""Baseline scheduling heuristics — paper §4.1."""

from .heuristics import (
    benchmark_assignment,
    first_fit_assignment,
    worst_fit_assignment,
)

__all__ = [
    "benchmark_assignment",
    "first_fit_assignment",
    "worst_fit_assignment",
]
