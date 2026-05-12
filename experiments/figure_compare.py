"""Baseline vs P1-improved comparison plots.

Reads:
    experiments/results/baseline_*.csv      — frozen pre-P1 snapshots
    experiments/results/*.csv               — post-P1 (current) data
    experiments/results/figure10_mobility.csv (P1-only, no baseline)

Writes paired plots into figs/:
    figs/compare_figure6_completion_time.png
    figs/compare_figure7_total_energy.png
    figs/compare_figure8_execution_energy.png
    figs/compare_figure9_success_rate.png
    figs/compare_figure10_mobility.png

P1 changes producing the deltas:
    一.1  time-varying distances via Topology.mobility (default mobility=None,
          so the *current* CSVs without mobility differ from baseline only via
          一.2/二.7 — Figure 10 is the dedicated mobility experiment).
    一.2  NHPP event timestamps (Event.time field; existing fixed-count
          generator still used, so Figure 9 numbers shift only via 二.7).
    二.7  Slack-Time Allocation replacing the ill-posed Eq. 14 in
          ``algorithm2._per_task_deadlines`` — directly changes Figure 9.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
FIGS = Path(__file__).resolve().parent.parent / "figs"
METHODS = ("UARP", "FF", "WF", "Benchmark")
MARKERS = {"UARP": "D", "FF": "^", "WF": "s", "Benchmark": "o"}
STYLES = {"UARP": "-", "FF": "--", "WF": "--", "Benchmark": ":"}


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def _compare_678(metric: str, ylabel: str, title: str, out_name: str) -> None:
    base = _read("baseline_figures678_compare.csv")
    new = _read("figures678_compare.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, df, label in zip(axes, (base, new), ("Baseline (pre-P1)", "P1 (SLF + mobility-aware)")):
        pivot = df.groupby(["scale", "method"], as_index=False)[metric].mean()
        pivot = pivot.pivot(index="scale", columns="method", values=metric)
        for m in METHODS:
            if m not in pivot.columns:
                continue
            ax.plot(pivot.index, pivot[m].values,
                    marker=MARKERS[m], linestyle=STYLES[m], label=m, linewidth=1.6)
        ax.set_xlabel("Number of tasks (M)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGS / out_name, dpi=150)
    plt.close(fig)


def _compare_fig9() -> None:
    base = _read("baseline_figure9_success_rate.csv")
    new = _read("figure9_success_rate.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, df, label in zip(axes, (base, new), ("Baseline (α·WT_ideal)", "P1 (Slack-Time Allocation)")):
        pivot = df.groupby(["alpha", "method"], as_index=False)["success_rate"].mean()
        pivot = pivot.pivot(index="alpha", columns="method", values="success_rate")
        for m in METHODS:
            if m not in pivot.columns:
                continue
            ax.plot(pivot.index, pivot[m].values,
                    marker=MARKERS[m], linestyle=STYLES[m], label=m, linewidth=1.6)
        ax.set_xlabel("Deadline coefficient α")
        ax.set_ylim(0, 1.05)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Success rate")
    axes[-1].legend(loc="best")
    fig.suptitle("Figure 9 comparison - success rate vs alpha (P1 #2.7 changes per-task DT)")
    fig.tight_layout()
    fig.savefig(FIGS / "compare_figure9_success_rate.png", dpi=150)
    plt.close(fig)


def _plot_fig10_alone() -> None:
    """Figure 10 has no baseline (mobility didn't exist before P1).

    We plot only the new curve, framed as the "post-P1" panel — paired with a
    visual reminder that v=0 corresponds to the static topology that the
    baseline Figure 9 already covered.
    """
    df = _read("figure10_mobility.csv")
    pivot = df.groupby(["velocity", "method"], as_index=False)["success_rate"].mean()
    pivot = pivot.pivot(index="velocity", columns="method", values="success_rate")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in METHODS:
        if m not in pivot.columns:
            continue
        ax.plot(pivot.index, pivot[m].values,
                marker=MARKERS[m], linestyle=STYLES[m], label=m, linewidth=1.6)
    ax.set_xlabel("Device velocity (distance-units / time-unit)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 10 - success rate vs mobility speed (P1 #1.1 only)")
    ax.axvline(0.0, color="grey", linestyle=":", alpha=0.5,
               label="v=0 ≡ static (matches baseline Figure 9 α=1.2)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "compare_figure10_mobility.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    _compare_678("completion_time", "Completion time (WT)",
                 "Figure 6 comparison — completion time", "compare_figure6_completion_time.png")
    _compare_678("total_energy", "Total energy (CM)",
                 "Figure 7 comparison — total energy", "compare_figure7_total_energy.png")
    _compare_678("execution_energy", "Execution energy (CME)",
                 "Figure 8 comparison — execution energy", "compare_figure8_execution_energy.png")
    _compare_fig9()
    _plot_fig10_alone()
    print("comparison plots written to figs/compare_figure*.png")


if __name__ == "__main__":
    main()
