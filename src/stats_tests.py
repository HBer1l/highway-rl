"""
=============================================================================
OVERKILL_FOR_UNDERGRAD_COURSE — REMOVE THIS WHOLE FILE / SECTION IF DESIRED
=============================================================================

Statistical comparison between V3a and V3c variants using both:
    - Welch's t-test (parametric, assumes approximately normal sample means)
    - Mann-Whitney U (non-parametric, no normality assumption)

With n=3 seeds these tests have very low statistical power, so a non-
significant result here means almost nothing. A significant result with
n=3 is suggestive but not definitive — proper claims would need n>=5 seeds.

This file is purely for methodological completeness. The bar-chart figure
and per-seed numbers in the README are sufficient for an undergrad project.

Usage:
    python -m src.stats_tests
=============================================================================
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats  # type: ignore

from .config import ASSETS_DIR


def main() -> None:
    csv_path = ASSETS_DIR / "multiseed_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python -m src.aggregate_seeds` first."
        )

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    v3a_crash = [float(r["crash_rate"]) for r in rows if r["label"] == "V3a"]
    v3c_crash = [float(r["crash_rate"]) for r in rows if r["label"] == "V3c"]
    v3a_ret   = [float(r["mean_return"]) for r in rows if r["label"] == "V3a"]
    v3c_ret   = [float(r["mean_return"]) for r in rows if r["label"] == "V3c"]

    print("=== V3a vs V3c — crash rate ===")
    print(f"  V3a: {np.mean(v3a_crash)*100:.1f} ± {np.std(v3a_crash, ddof=1)*100:.1f}% "
          f"(n={len(v3a_crash)})")
    print(f"  V3c: {np.mean(v3c_crash)*100:.1f} ± {np.std(v3c_crash, ddof=1)*100:.1f}% "
          f"(n={len(v3c_crash)})")

    t_stat, t_p = stats.ttest_ind(v3a_crash, v3c_crash, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(v3a_crash, v3c_crash, alternative="greater")
    print(f"  Welch's t-test:   t={t_stat:.3f}, p={t_p:.4f}")
    print(f"  Mann-Whitney U:   U={u_stat:.1f}, p={u_p:.4f} (one-sided: V3a > V3c)")

    print("\n=== V3a vs V3c — return ===")
    print(f"  V3a: {np.mean(v3a_ret):.2f} ± {np.std(v3a_ret, ddof=1):.2f}")
    print(f"  V3c: {np.mean(v3c_ret):.2f} ± {np.std(v3c_ret, ddof=1):.2f}")
    t_stat, t_p = stats.ttest_ind(v3a_ret, v3c_ret, equal_var=False)
    print(f"  Welch's t-test:   t={t_stat:.3f}, p={t_p:.4f}")

    print("\nNote: n=3 per group. p-values are suggestive, not definitive.")


if __name__ == "__main__":
    main()
