"""
Generate publication-quality reward curves from logs/<version>/rewards.csv.

Two figures are produced:
    assets/reward_curve.png      – the main training curve for V3_FINAL
    assets/reward_comparison.png – V1 vs V3, demonstrating the impact of
                                   reward shaping iterations

Usage:
    python -m src.plot_rewards
    python -m src.plot_rewards --compare       # also build comparison plot
                                                  (requires V1 logs to exist)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import ASSETS_DIR, LOG_DIR, RewardVersion
from .utils import smooth


# Use a clean, paper-quality style — no seaborn dependency
plt.rcParams.update(
    {
        "figure.figsize": (10, 5),
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "font.size": 11,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training reward curves.")
    parser.add_argument("--compare", action="store_true", help="Also build V1 vs V3 plot.")
    parser.add_argument("--smooth-window", type=int, default=20)
    return parser.parse_args()


def _load_rewards(version: RewardVersion) -> tuple[np.ndarray, np.ndarray]:
    """Load (timesteps, rewards) arrays from the per-version CSV log."""
    path = LOG_DIR / version.value / "rewards.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No log found at {path}. Train with `--reward {version.value}` first."
        )
    timesteps: list[int] = []
    rewards: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            timesteps.append(int(row["timestep"]))
            rewards.append(float(row["reward"]))
    return np.asarray(timesteps), np.asarray(rewards)


def plot_main_curve(window: int = 20) -> Path:
    ts, rewards = _load_rewards(RewardVersion.V3_FINAL)
    smoothed = smooth(rewards, window=window)

    fig, ax = plt.subplots()
    ax.plot(ts, rewards, color="#9ec5ff", alpha=0.4, label="raw episode reward")
    ax.plot(ts, smoothed, color="#1f77b4", linewidth=2.0, label=f"moving avg (n={window})")

    # Annotate the half-trained checkpoint
    ax.axvline(50_000, color="#666", linestyle=":", linewidth=1)
    ax.text(
        50_000,
        ax.get_ylim()[1] * 0.95,
        " halftrained\n checkpoint",
        fontsize=9,
        color="#444",
        verticalalignment="top",
    )

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode reward")
    ax.set_title("PPO training on merge-v0 (V3 reward)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()

    out_path = ASSETS_DIR / "reward_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")
    return out_path


def plot_comparison(window: int = 20) -> Path:
    """Compare V1 (naive) and V3 (final) reward shaping side by side."""
    ts_v1, r_v1 = _load_rewards(RewardVersion.V1_NAIVE)
    ts_v3, r_v3 = _load_rewards(RewardVersion.V3_FINAL)

    fig, ax = plt.subplots()
    ax.plot(ts_v1, smooth(r_v1, window), color="#d62728", linewidth=2.0, label="V1 (naive)")
    ax.plot(ts_v3, smooth(r_v3, window), color="#2ca02c", linewidth=2.0, label="V3 (final)")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Smoothed episode reward")
    ax.set_title("Reward shaping ablation: V1 vs V3")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()

    out_path = ASSETS_DIR / "reward_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    plot_main_curve(window=args.smooth_window)
    if args.compare:
        plot_comparison(window=args.smooth_window)


if __name__ == "__main__":
    main()
