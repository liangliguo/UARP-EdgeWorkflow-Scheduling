"""Reproduce all figures in one go.

Usage:
    uv run python -m experiments.run_all
"""

from __future__ import annotations

import time

from . import figure5_utility, figure9_success_rate, figures678_compare


def main() -> None:
    t0 = time.time()
    print("=== Figure 5: UARP utility ===")
    figure5_utility.main()
    print(f"  done in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("=== Figures 6/7/8: comparison ===")
    figures678_compare.main()
    print(f"  done in {time.time() - t1:.1f}s")

    t2 = time.time()
    print("=== Figure 9: success rate ===")
    figure9_success_rate.main()
    print(f"  done in {time.time() - t2:.1f}s")

    print(f"all figures regenerated in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
