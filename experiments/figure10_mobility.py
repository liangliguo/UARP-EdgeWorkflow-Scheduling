"""Figure 10 — success rate vs device velocity (IMPROVEMENTS.md 一.1).

For each velocity v ∈ {0, 1, 2, 4, 8, 16} we build a heterogeneous topology
with a ``linear_walk`` mobility trace (per-node random direction) and measure
UARP / FF / WF / Benchmark success rates at α=1.2. ``v=0`` is the static
baseline; ``v>0`` shows how each method degrades as the device moves.

The point is that UARP only "sees" mobility through ``schedule_times`` —
no method here is yet *mobility-aware* in the optimiser. Future P4 work
would feed predicted distances into the NSGA evaluation; this figure is
the reference curve to beat.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
    linear_walk,
    reschedule,
    reschedule_benchmark,
    uarp_sub_scheduler,
    wf_sub_scheduler,
)

from .config import CFG
from .figures678_compare import METHODS


SCALE_FOR_FIG10 = 25
ALPHA_FOR_FIG10 = 1.2
T_HORIZON = 200.0  # mobility window in time-units (matches max workflow WT)
VELOCITY_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _make_topo(seed: int, velocity: float):
    topo = make_heterogeneous_topology(N=CFG.n_edge_nodes, BA=CFG.BA, seed=seed)
    if velocity > 0.0:
        topo.mobility = linear_walk(
            base_distances=topo.distances,
            velocity=velocity,
            T_horizon=T_HORIZON,
            seed=seed,
        )
    return topo


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


def _run(method, wf, topo, init, events, alpha, seed, dt_ref):
    common = dict(
        alpha=alpha, progress_frac=CFG.progress_frac, deadline_reference=dt_ref,
    )
    if method == "UARP":
        return reschedule(
            wf, topo, init, events,
            uarp_sub_scheduler(pop_size=CFG.pop_size // 2, n_gen=CFG.n_gen // 2, seed=seed),
            **common,
        )
    if method == "FF":
        return reschedule(wf, topo, init, events, ff_sub_scheduler(), **common)
    if method == "WF":
        return reschedule(wf, topo, init, events, wf_sub_scheduler(), **common)
    return reschedule_benchmark(wf, topo, init, events, **common)


def run() -> pd.DataFrame:
    rows = []
    for repeat in range(CFG.n_repeats):
        seed = CFG.seed + repeat
        wf = random_dag(
            n_tasks=SCALE_FOR_FIG10, edge_prob=CFG.edge_prob,
            size_range=CFG.size_range, seed=seed,
        )
        for v in VELOCITY_GRID:
            topo = _make_topo(seed=seed, velocity=v)
            events = generate_events(topo, n_events=CFG.n_uncertainty_events, seed=seed)
            dt_ref = _initial("UARP", wf, topo, seed=seed)
            for method in METHODS:
                init = _initial(method, wf, topo, seed=seed)
                res = _run(method, wf, topo, init, events, ALPHA_FOR_FIG10, seed, dt_ref)
                suc = success_rate(res.affected_finishes, res.affected_deadlines)
                rows.append(
                    {
                        "velocity": v,
                        "method": method,
                        "repeat": repeat,
                        "success_rate": suc,
                    }
                )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path) -> None:
    grouped = df.groupby(["velocity", "method"], as_index=False)["success_rate"].mean()
    pivot = grouped.pivot(index="velocity", columns="method", values="success_rate")
    pivot = pivot[list(METHODS)]
    markers = {"UARP": "D", "FF": "^", "WF": "s", "Benchmark": "o"}
    styles = {"UARP": "-", "FF": "--", "WF": "--", "Benchmark": ":"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in METHODS:
        ax.plot(
            pivot.index, pivot[m].values,
            marker=markers[m], linestyle=styles[m], label=m, linewidth=1.6,
        )
    ax.set_xlabel("Device velocity (distance-units / time-unit)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Figure 10 — Success rate vs mobility speed (α={ALPHA_FOR_FIG10})")
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
    df.to_csv(out_dir / "figure10_mobility.csv", index=False)
    plot(df, figs_dir / "figure10_mobility.png")
    print(df.groupby(["velocity", "method"])["success_rate"].mean().unstack().to_string())


if __name__ == "__main__":
    main()
