"""Figure 5 — UARP utility values across Pareto solutions for 6 task scales.

Paper §4.2: for each workflow scale (10/15/20/25/30/35), run NSGA-III, compute
the SAW utility of each Pareto solution, and bar-chart them. The chosen
strategy is the bar with the highest utility.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uarp.model import make_heterogeneous_topology, random_dag
from uarp.scheduler import solve

from .config import CFG


def run() -> pd.DataFrame:
    rows: list[dict] = []
    for scale in CFG.task_scales:
        wf = random_dag(
            n_tasks=scale, edge_prob=CFG.edge_prob, size_range=CFG.size_range, seed=CFG.seed
        )
        topo = make_heterogeneous_topology(
            N=CFG.n_edge_nodes,
            BA=CFG.BA,
            seed=CFG.seed,
        )
        res = solve(
            wf,
            topo,
            pop_size=CFG.pop_size,
            n_gen=CFG.n_gen,
            n_partitions=CFG.n_partitions,
            seed=CFG.seed,
        )
        for i, u in enumerate(res.utilities):
            rows.append({"scale": scale, "solution": i + 1, "utility": float(u)})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path) -> None:
    scales = list(CFG.task_scales)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    panel_letters = ["A", "B", "C", "D", "E", "F"]
    for ax, scale, letter in zip(axes.flat, scales, panel_letters):
        sub = df[df["scale"] == scale]
        ax.bar(sub["solution"].astype(str), sub["utility"])
        ax.set_title(f"({letter}) tasks = {scale}")
        ax.set_xlabel("solution")
        ax.set_ylabel("Utility value")
        ax.set_ylim(0, 1.0)
        if not sub.empty:
            best = sub.loc[sub["utility"].idxmax(), "solution"]
            ax.bar([str(best)], [sub["utility"].max()], color="C3")
    fig.suptitle("Figure 5 — UARP Pareto utility values per workflow scale")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    df = run()
    df.to_csv(out_dir / "figure5_utility.csv", index=False)
    plot(df, Path(__file__).resolve().parent.parent / "figs" / "figure5_utility.png")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
