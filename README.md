# UARP — paper reproduction (CCPE 2020, e5674)

Reproduction of *Dynamic resource provisioning for workflow scheduling under uncertainty in edge computing environment* (Xu, Cao, Geng, Liu, Dai, Wang). DOI: [10.1002/cpe.5674](https://doi.org/10.1002/cpe.5674).

## Quick start

```bash
uv sync
uv run pytest tests/                      # 29 unit tests
uv run python -m experiments.run_all      # regenerate Figures 5/6/7/8/9
```

Outputs:
- `experiments/results/*.csv` — raw metrics
- `figs/*.png` — Figures 5–9

## Layout

| Path | Purpose |
|---|---|
| `src/uarp/model/` | DAG workflow, edge nodes, schedule, cost formulas (Eqs. 4–16) |
| `src/uarp/scheduler/` | NSGA-III (pymoo) + SAW/MCDM (Eqs. 19–23), Algorithm 1 |
| `src/uarp/uncertainty/` | Events + Algorithm 2 dynamic resource provisioning |
| `src/uarp/baselines/` | Benchmark / First Fit / Worst Fit (paper §4.1) |
| `experiments/` | Per-figure reproduction scripts |
| `tests/` | Hand-traced unit tests for every formula |

## Reproduction notes (deviations from paper)

The paper does not publish source code or several experimental knobs. The
following choices are ours and live in `experiments/config.py`:

| Knob | Paper | Reproduction |
|---|---|---|
| Workflow DAG | unspecified | random DAG, edge prob 0.3, sizes 50–200 |
| Node heterogeneity | unspecified | capacity ∈ [1500, 2500] MHz, ce ∈ [0.04, 0.07] |
| Distances | unspecified | uniform(10, 100) |
| ct(v, z), ce(z), cs(i,j) | unspecified | size·distance·1e-5, ce per node, 1e-3 const |
| NSGA-III pop / generations | unspecified | 100 / 100 (Das–Dennis ref-dirs) |
| Repeats per (scale, method) | unspecified | 5 |
| Uncertainty event count | unspecified | 1 per workflow |
| Progress fraction at uncertainty | unspecified | 0.4 |

Other deviations:

1. **Service failure semantics** — modelled as severe degradation
   (`capacity *= 0.05`) rather than complete unavailability. This avoids
   `inf` completion times when the Benchmark method (which never reschedules)
   is assigned to a failed node, which matches paper §4.1(1)'s description
   of Benchmark as "may exceed the deadline" (not "fail outright").

2. **Per-task deadlines (Eq. 14)** — used a single shared reference
   schedule (UARP's ideal/uncertainty-free plan) across all comparison
   methods. Otherwise each method's deadline scales with its own initial
   schedule, which makes Benchmark trivially satisfy its loose deadline and
   inverts the paper's Figure 9 trend.

3. **Algorithm 1** — implemented via pymoo's `NSGA3` (mathematically
   equivalent to paper's Algorithm 1 lines 4–18). Crossover = SBX with
   RoundingRepair (integer chromosomes), mutation = Polynomial Mutation.

## Reproduced trends

- **Figure 6/7/8 (completion time / total energy / execution energy)**:
  UARP < FF / WF / Benchmark across all scales 10–35 — matches paper.
- **Figure 9 (success rate vs α)**: UARP rises 0.51 → 1.00 over α ∈
  [1.1, 1.4]; baselines stay near zero. Paper shows the same ordering and
  convergence direction; the absolute spread is larger here because our
  random DAG + heterogeneous topology produces more punishing tail tasks for
  the unaware FF/WF baselines.

## See also
- `TODO.md` — phase-by-phase reproduction plan
- `notation.md` — symbol table mapping paper notation ↔ code
- `ref/xu2020.pdf` — the paper itself
