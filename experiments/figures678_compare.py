"""Figures 6/7/8 — completion time, total energy, execution energy.

For each task scale ∈ {10, 15, 20, 25, 30, 35} we run all four methods
(UARP / FF / WF / Benchmark) under the same uncertainty event sequence and
average the metric over CFG.n_repeats workflow seeds.

Each method uses Algorithm 2's flow on the post-event topology so the
comparison is fair (Benchmark = no rescheduling, others = reschedule).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uarp.baselines import (
    benchmark_assignment,
    first_fit_assignment,
    worst_fit_assignment,
)
from uarp.model import (
    Schedule,
    completion_time,
    execution_energy,
    make_heterogeneous_topology,
    random_dag,
    total_energy,
)
from uarp.scheduler import solve as uarp_solve
from uarp.uncertainty import (
    ff_sub_scheduler,
    generate_events,
    reschedule,
    reschedule_benchmark,
    uarp_sub_scheduler,
    wf_sub_scheduler,
)

from .config import CFG


METHODS = ("UARP", "FF", "WF", "Benchmark")


def _initial_schedule(method: str, wf, topo, seed: int) -> Schedule:
    if method == "UARP":
        return uarp_solve(
            wf, topo,
            pop_size=CFG.pop_size,
            n_gen=CFG.n_gen,
            n_partitions=CFG.n_partitions,
            seed=seed,
        ).best
    if method == "FF":
        return first_fit_assignment(wf, topo)
    if method == "WF":
        return worst_fit_assignment(wf, topo)
    if method == "Benchmark":
        return benchmark_assignment(wf, topo, seed=seed)
    raise ValueError(method)


def _run_with_uncertainty(method: str, wf, topo, init: Schedule, events, *, seed: int):
    if method == "UARP":
        return reschedule(
            wf, topo, init, events,
            uarp_sub_scheduler(pop_size=CFG.pop_size // 2, n_gen=CFG.n_gen // 2, seed=seed),
            alpha=1.2, progress_frac=CFG.progress_frac,
        )
    if method == "FF":
        return reschedule(
            wf, topo, init, events, ff_sub_scheduler(),
            alpha=1.2, progress_frac=CFG.progress_frac,
        )
    if method == "WF":
        return reschedule(
            wf, topo, init, events, wf_sub_scheduler(),
            alpha=1.2, progress_frac=CFG.progress_frac,
        )
    if method == "Benchmark":
        return reschedule_benchmark(
            wf, topo, init, events,
            alpha=1.2, progress_frac=CFG.progress_frac,
        )
    raise ValueError(method)


def run() -> pd.DataFrame:
    rows: list[dict] = []
    for scale in CFG.task_scales:
        for repeat in range(CFG.n_repeats):
            seed = CFG.seed + scale * 100 + repeat
            wf = random_dag(
                n_tasks=scale, edge_prob=CFG.edge_prob,
                size_range=CFG.size_range, seed=seed,
            )
            topo = make_heterogeneous_topology(N=CFG.n_edge_nodes, BA=CFG.BA, seed=seed)
            events = generate_events(topo, n_events=CFG.n_uncertainty_events, seed=seed)
            for method in METHODS:
                init = _initial_schedule(method, wf, topo, seed=seed)
                res = _run_with_uncertainty(method, wf, topo, init, events, seed=seed)
                # Use the post-event topology for execution-energy reporting
                cme = execution_energy(wf, res.final_topology, res.final_schedule)
                rows.append(
                    {
                        "scale": scale,
                        "repeat": repeat,
                        "method": method,
                        "completion_time": res.actual_wt,
                        "total_energy": res.actual_cm,
                        "execution_energy": cme,
                    }
                )
    return pd.DataFrame(rows)


def _bar_chart(df: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: Path):
    grouped = df.groupby(["scale", "method"], as_index=False)[metric].mean()
    pivot = grouped.pivot(index="scale", columns="method", values=metric)
    pivot = pivot[list(METHODS)]  # column order
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.2
    x = np.arange(len(pivot.index))
    for i, m in enumerate(METHODS):
        ax.bar(x + (i - 1.5) * width, pivot[m].values, width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_xlabel("The number of tasks")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    figs_dir = Path(__file__).resolve().parent.parent / "figs"
    df = run()
    df.to_csv(out_dir / "figures678_compare.csv", index=False)
    _bar_chart(
        df, "completion_time", "Completion time (s)",
        "Figure 6 — Completion time", figs_dir / "figure6_completion_time.png",
    )
    _bar_chart(
        df, "total_energy", "Total energy (kW)",
        "Figure 7 — Total energy consumption", figs_dir / "figure7_total_energy.png",
    )
    _bar_chart(
        df, "execution_energy", "Execution energy (kW)",
        "Figure 8 — Execution energy consumption", figs_dir / "figure8_execution_energy.png",
    )
    summary = df.groupby(["scale", "method"], as_index=False)[
        ["completion_time", "total_energy", "execution_energy"]
    ].mean()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
