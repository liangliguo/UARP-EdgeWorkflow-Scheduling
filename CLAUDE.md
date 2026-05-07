# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Reproduction of Xu et al. 2020 (CCPE e5674, DOI 10.1002/cpe.5674) — *Dynamic resource provisioning for workflow scheduling under uncertainty in edge computing environment*. The paper is in `ref/xu2020.pdf`. The method (UARP) schedules a DAG workflow onto edge nodes by treating it as a 2-objective problem (completion time + energy) solved with NSGA-III, then handles uncertainty (node degradation, service failure, new node joins) by rescheduling the un-finished sub-DAG on the post-event topology.

`README.md` lists every reproduction-time choice that is **not** specified in the paper (random DAG params, NSGA-III hyper-params, ce/ct/cs coefficients, etc.). Read it before adding experiments — these defaults live in `experiments/config.py`.

`notation.md` is the symbol bridge between the paper (V, M, N, x_{m,i,k}, WT, CM, DT, SUC, etc.) and the code (`Workflow.tasks`, `Schedule.assignment`, `cost.completion_time`, …). When discussing equations, cite them as `Eq. N` matching the paper's numbering — that numbering is referenced throughout the source.

## Commands

```bash
uv sync                                      # install/refresh deps
uv run pytest tests/                         # all 29 unit tests
uv run pytest tests/test_model.py -v         # one file
uv run pytest tests/test_model.py::test_eq4_transmission_time  # one test
uv run python -m experiments.run_all         # regenerate all 5 figures + CSVs (~45s)
uv run python -m experiments.figure5_utility # one figure at a time
uv run ruff check src tests experiments      # lint
```

The package is installed in editable mode via `[tool.hatch.build]` pointing at `src/uarp`. Tests rely on `pythonpath = ["src"]` in `pyproject.toml` — run them with `uv run pytest`, not bare `pytest`.

## Architecture

The pipeline mirrors the paper's section structure. Each layer is a separate package under `src/uarp/`; cross-layer code goes through the public APIs in each `__init__.py`.

```
model/      — paper §2: Workflow (DAG), EdgeNode/Topology, Schedule, cost (Eqs. 4–16)
scheduler/  — paper §3: pymoo NSGA-III problem + Algorithm 1 + SAW/MCDM (Eqs. 19–23)
baselines/  — paper §4.1: Benchmark / First Fit / Worst Fit
uncertainty/— paper §3.3–3.4: events + Algorithm 2 (sub-DAG rescheduling)
experiments/— paper §4.2–4.3: per-figure scripts → figs/*.png + experiments/results/*.csv
```

### Key contracts

- **Schedule.assignment** is an integer ndarray of length M, values in `[0, N-1]`. Every cost function takes `(workflow, topology, schedule)` and returns a scalar — they're vectorised over the workflow internally but called per-individual by NSGA-III.
- **`cost.schedule_times()`** returns `(ST, WT)` arrays of length M by walking `workflow.topo_order()`. A task starts when *all* predecessors have finished AND its data has been transmitted to its assigned node — not the literal Eq. 7/8 (which is ambiguous in the PDF). The unit tests in `test_model.py::test_serial_schedule_times` and `test_parallel_schedule_times` lock the chosen interpretation.
- **`uarp.scheduler.solve()`** is the entry point for both Algorithm 1 (initial scheduling) and Algorithm 2's rescheduling step. The `UARPResult` it returns carries the full Pareto front *and* the SAW-chosen `Schedule` — Figure 5 visualises the front; Figures 6/7/8 use only `.best`.
- **`uncertainty.reschedule()`** has a pluggable `sub_scheduler: (Workflow, Topology) -> Schedule`. UARP, FF, WF all share this skeleton — only the inner heuristic differs. `reschedule_benchmark()` is separate because Benchmark, by definition, does not reschedule.
- **Per-task deadlines (Eq. 14)** must use a *single* shared reference schedule across all comparison methods (we pass UARP's ideal/uncertainty-free plan as `deadline_reference`). Otherwise Benchmark trivially passes its own loose deadline and Figure 9's trend inverts. This is the most subtle reproduction trap — see `algorithm2.py::_per_task_deadlines` and the docstring.
- **`service_failure` events** apply as severe degradation (`capacity *= 0.05`), not `available = False`. This keeps Benchmark "may exceed deadline" semantics (paper §4.1(1)) instead of producing `inf` completion times. `EdgeNode.available` is still threaded through the model in case stricter semantics are needed later.

### Adding a new method or experiment

1. New scheduling heuristic → add to `src/uarp/baselines/` and re-export from its `__init__.py`. Its initial-assignment function must take `(Workflow, Topology, *, seed=None)` and return a `Schedule`; for uncertainty support, also expose a `sub_scheduler()` factory in the same shape as `ff_sub_scheduler` / `wf_sub_scheduler` and add a branch in `experiments/figures678_compare._initial_schedule`.
2. New experiment / figure → add `experiments/figureN_*.py` with a `run() -> pd.DataFrame` and a `main()` that writes the CSV under `experiments/results/` and the PNG under `figs/`. Wire it into `experiments/run_all.py`. Use `experiments.config.CFG` for any tunable — never hard-code.
3. New cost term or formula → put it in `src/uarp/model/cost.py` with its paper-equation reference in the docstring, re-export from `model/__init__.py`, and add a hand-traced test in `tests/test_model.py` like the existing Eq-numbered tests.

### Pitfalls observed during reproduction

- **Pymoo emits floats for integer chromosomes.** `WorkflowSchedulingProblem` declares `vtype=int`, but pymoo still passes float arrays into `_evaluate`; the SBX crossover and PM mutation are wrapped with `RoundingRepair()`. Don't remove that wrapper or you'll get out-of-range node indices.
- **Homogeneous edge nodes collapse the Pareto front.** When all nodes share capacity/ce, minimising WT ≡ minimising CM, so NSGA-III returns 1 solution. Use `make_heterogeneous_topology` (default in experiments) for any optimisation comparison; reserve `make_homogeneous_topology` for unit tests where the trade-off doesn't matter.
- **`Workflow.subset()`** returns a *local-indexed* sub-workflow plus a `global_to_local` ordering list. Algorithm 2 stitches the new schedule back into the original M-length assignment using that mapping — don't lose track of which index space you're in.
