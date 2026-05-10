"""
Multi-seed aggregation: evaluate every (variant, seed) checkpoint over N
episodes and produce a comparison table + bar chart with error bars.

Usage:
    python -m src.aggregate_seeds                          # use defaults
    python -m src.aggregate_seeds --episodes 100           # more eval episodes

Outputs:
    assets/multiseed_comparison.png  – grouped bar chart (return, length, crash)
    assets/multiseed_table.csv       – full numeric table
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import ASSETS_DIR, CHECKPOINT_DIR, CONFIG, RewardVersion
from .evaluate import evaluate


# --- registry of (variant, seed, checkpoint_filename) ----------------------
@dataclass(frozen=True)
class Run:
    label: str
    reward_version: RewardVersion
    seed: int
    checkpoint: str


RUNS: list[Run] = [
    Run("V3a", RewardVersion.V3_FINAL, 42,  "fulltrained_v3a.zip"),
    Run("V3a", RewardVersion.V3_FINAL, 7,   "v3_final_seed7_final.zip"),
    Run("V3a", RewardVersion.V3_FINAL, 123, "v3_final_seed123_final.zip"),
    Run("V3c", RewardVersion.V3_TTC,   42,  "fulltrained_v3c_seed42.zip"),
    Run("V3c", RewardVersion.V3_TTC,   7,   "v3_ttc_seed7_final.zip"),
    Run("V3c", RewardVersion.V3_TTC,   123, "v3_ttc_seed123_final.zip"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for run in RUNS:
        ckpt = CHECKPOINT_DIR / run.checkpoint
        if not ckpt.exists():
            print(f"[skip] {ckpt} missing")
            continue
        stats = evaluate(
            cfg=CONFIG,
            checkpoint=ckpt,
            reward_version=run.reward_version,
            n_episodes=args.episodes,
            record_path=None,
            seed=2024,
        )
        rows.append({
            "label": run.label,
            "seed": run.seed,
            "checkpoint": run.checkpoint,
            **stats,
        })

    # --- write CSV
    out_csv = ASSETS_DIR / "multiseed_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {out_csv}")

    # --- aggregate per variant
    variants = sorted({r["label"] for r in rows})
    agg = {}
    for v in variants:
        sub = [r for r in rows if r["label"] == v]
        agg[v] = {
            "crash_rate":  ([r["crash_rate"]  for r in sub]),
            "mean_return": ([r["mean_return"] for r in sub]),
            "mean_length": ([r["mean_length"] for r in sub]),
        }

    # --- plot grouped bar chart with error bars
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = [
        ("crash_rate", "Crash rate", lambda x: x * 100, "%"),
        ("mean_return", "Mean episode return", lambda x: x, ""),
        ("mean_length", "Mean episode length", lambda x: x, "steps"),
    ]
    colors = {"V3a": "#d62728", "V3c": "#2ca02c"}

    for ax, (key, title, transform, unit) in zip(axes, metrics):
        means = []
        stds = []
        labels = []
        bar_colors = []
        for v in variants:
            vals = [transform(x) for x in agg[v][key]]
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            labels.append(v)
            bar_colors.append(colors.get(v, "#888"))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=8, color=bar_colors, alpha=0.85,
               edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        # annotate values
        for xi, m, s in zip(x, means, stds):
            ax.text(xi, m + s + (max(means) * 0.02), f"{m:.1f}",
                    ha="center", fontsize=9)

    fig.suptitle("Multi-seed comparison (n=3 seeds × 50 eval episodes per seed)",
                 fontsize=12)
    fig.tight_layout()

    out_png = ASSETS_DIR / "multiseed_comparison.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[saved] {out_png}")

    # --- print summary
    print("\n=== Summary (mean ± std across seeds) ===")
    for v in variants:
        cr = np.array(agg[v]["crash_rate"]) * 100
        ret = np.array(agg[v]["mean_return"])
        ln = np.array(agg[v]["mean_length"])
        print(f"  {v}: crash={cr.mean():.1f} ± {cr.std(ddof=1):.1f}% | "
              f"return={ret.mean():.2f} ± {ret.std(ddof=1):.2f} | "
              f"length={ln.mean():.1f} ± {ln.std(ddof=1):.1f}")


if __name__ == "__main__":
    main()
