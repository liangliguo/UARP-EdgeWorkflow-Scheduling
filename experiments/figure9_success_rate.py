"""Figure 9 — success rate vs deadline coefficient α ∈ {1.1, 1.2, 1.3, 1.4}.

Success rate (Eq. 16): fraction of affected tasks finishing before their
per-task deadline. We average over CFG.n_repeats workflows at a fixed scale
(use the largest scale = 35 to mirror the paper's setup, where success rate
matters most when there are many tasks and tighter coupling).
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
    make_heterogeneous_topology,
    random_dag,
    success_rate,
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
from .figures678_compare import METHODS


SCALE_FOR_FIG9 = 25  # one scale, varied α


def _initial(method: str, wf, topo, seed: int) -> Schedule:
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
    return benchmark_assignment(wf, topo, seed=seed)


def _run(method: str, wf, topo, init, events, alpha, seed, dt_ref):
    if method == "UARP":
        return reschedule(
            wf, topo, init, events,
            uarp_sub_scheduler(pop_size=CFG.pop_size // 2, n_gen=CFG.n_gen // 2, seed=seed),
            alpha=alpha, progress_frac=CFG.progress_frac,
            deadline_reference=dt_ref,
        )
    if method == "FF":
        return reschedule(
            wf, topo, init, events, ff_sub_scheduler(),
            alpha=alpha, progress_frac=CFG.progress_frac,
            deadline_reference=dt_ref,
        )
    if method == "WF":
        return reschedule(
            wf, topo, init, events, wf_sub_scheduler(),
            alpha=alpha, progress_frac=CFG.progress_frac,
            deadline_reference=dt_ref,
        )
    return reschedule_benchmark(
        wf, topo, init, events,
        alpha=alpha, progress_frac=CFG.progress_frac,
        deadline_reference=dt_ref,
    )


def run() -> pd.DataFrame:
    rows: list[dict] = []
    for repeat in range(CFG.n_repeats):
        seed = CFG.seed + repeat
        wf = random_dag(
            n_tasks=SCALE_FOR_FIG9, edge_prob=CFG.edge_prob,
            size_range=CFG.size_range, seed=seed,
        )
        topo = make_heterogeneous_topology(N=CFG.n_edge_nodes, BA=CFG.BA, seed=seed)
        events = generate_events(topo, n_events=CFG.n_uncertainty_events, seed=seed)
        # shared deadline reference: UARP's ideal (uncertainty-free) schedule
        dt_ref = _initial("UARP", wf, topo, seed=seed)
        for method in METHODS:
            init = _initial(method, wf, topo, seed=seed)
            for alpha in CFG.alpha_grid:
                res = _run(method, wf, topo, init, events, alpha, seed, dt_ref)
                suc = success_rate(res.affected_finishes, res.affected_deadlines)
                rows.append(
                    {
                        "alpha": alpha,
                        "method": method,
                        "repeat": repeat,
                        "success_rate": suc,
                    }
                )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path) -> None:
    grouped = df.groupby(["alpha", "method"], as_index=False)["success_rate"].mean()
    pivot = grouped.pivot(index="alpha", columns="method", values="success_rate")
    pivot = pivot[list(METHODS)]
    markers = {"UARP": "D", "FF": "^", "WF": "s", "Benchmark": "o"}
    styles = {"UARP": "-", "FF": "--", "WF": "--", "Benchmark": ":"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in METHODS:
        ax.plot(
            pivot.index, pivot[m].values,
            marker=markers[m], linestyle=styles[m], label=m, linewidth=1.6,
        )
    ax.set_xlabel("Deadline variable α")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Figure 9 — Success rate vs α (tasks={SCALE_FOR_FIG9})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    figs_dir = Path(__file__).resolve().parent.parent / "figs"
    figs_dir.mkdir(exist_ok=True)
    df = run()
    df.to_csv(out_dir / "figure9_success_rate.csv", index=False)
    plot(df, figs_dir / "figure9_success_rate.png")
    print(
        df.groupby(["alpha", "method"])["success_rate"].mean().unstack().to_string()
    )


if __name__ == "__main__":
    main()
