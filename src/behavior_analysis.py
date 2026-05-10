"""
Behavioral analysis: roll out trained policies for many episodes, record
per-step quantities, and produce distribution plots that show how V3a and
V3c differ qualitatively (not just by crash rate).

Outputs assets/behavior_distributions.png with four panels:
    - Episode length histogram
    - Mean per-episode speed histogram
    - Lane-index over time (line plot, one trace per episode)
    - Action distribution (bar chart of action frequencies)

Usage:
    python -m src.behavior_analysis
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from .config import ASSETS_DIR, CHECKPOINT_DIR, CONFIG, RewardVersion
from .env_wrapper import make_env

# Best policy per variant — picked from multi-seed eval, not "production" file
TARGETS: list[tuple[str, RewardVersion, str]] = [
    ("V3a (β=1.0)",  RewardVersion.V3_FINAL, "fulltrained_v3a.zip"),
    ("V3c (TTC)",    RewardVersion.V3_TTC,   "fulltrained.zip"),
]
ACTION_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=4242)
    return p.parse_args()


def collect(label: str, version: RewardVersion, ckpt: str, n_eps: int, base_seed: int):
    model = PPO.load(str(CHECKPOINT_DIR / ckpt))
    env = make_env(CONFIG.env, version, seed=base_seed)
    lengths, speeds, actions = [], [], []
    lane_traces = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=base_seed + ep)
        ep_speeds = []
        ep_lanes = []
        ep_len = 0
        done = False
        while not done:
            ego = env.unwrapped.vehicle
            if ego is not None:
                ep_speeds.append(float(np.linalg.norm(ego.velocity)))
                try:
                    ep_lanes.append(int(ego.lane_index[2]))
                except Exception:
                    ep_lanes.append(0)
            action, _ = model.predict(obs, deterministic=True)
            actions.append(int(action))
            obs, _r, term, trunc, _info = env.step(action)
            ep_len += 1
            done = bool(term or trunc)
        lengths.append(ep_len)
        if ep_speeds:
            speeds.append(float(np.mean(ep_speeds)))
        if ep_lanes:
            lane_traces.append(ep_lanes)
    env.close()
    return {
        "label": label,
        "lengths": lengths,
        "speeds": speeds,
        "actions": actions,
        "lane_traces": lane_traces,
    }


def main() -> None:
    args = parse_args()
    data = [collect(lbl, ver, ck, args.episodes, args.seed) for lbl, ver, ck in TARGETS]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#d62728", "#2ca02c"]

    # 1. Episode length histogram
    ax = axes[0, 0]
    for d, c in zip(data, colors):
        ax.hist(d["lengths"], bins=15, alpha=0.6, label=d["label"], color=c, edgecolor="black")
    ax.set_xlabel("Episode length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Survival distribution")
    ax.legend(frameon=False)

    # 2. Mean episode speed histogram
    ax = axes[0, 1]
    for d, c in zip(data, colors):
        ax.hist(d["speeds"], bins=15, alpha=0.6, label=d["label"], color=c, edgecolor="black")
    ax.set_xlabel("Mean speed per episode (m/s)")
    ax.set_ylabel("Count")
    ax.set_title("Speed distribution")
    ax.legend(frameon=False)

    # 3. Lane index over time (mean ± band across episodes)
    ax = axes[1, 0]
    for d, c in zip(data, colors):
        # Pad/truncate traces to common length and average
        max_len = max((len(t) for t in d["lane_traces"]), default=0)
        if max_len == 0:
            continue
        padded = np.full((len(d["lane_traces"]), max_len), np.nan)
        for i, t in enumerate(d["lane_traces"]):
            padded[i, :len(t)] = t
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        x = np.arange(max_len)
        ax.plot(x, mean, color=c, label=d["label"], linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.2)
    ax.set_xlabel("Step within episode")
    ax.set_ylabel("Lane index")
    ax.set_title("Lane position over time (mean ± std)")
    ax.legend(frameon=False)

    # 4. Action distribution
    ax = axes[1, 1]
    width = 0.4
    x = np.arange(len(ACTION_NAMES))
    for i, (d, c) in enumerate(zip(data, colors)):
        counts = Counter(d["actions"])
        total = sum(counts.values()) or 1
        freqs = [counts.get(a, 0) / total for a in range(len(ACTION_NAMES))]
        ax.bar(x + (i - 0.5) * width, freqs, width, color=c, label=d["label"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_NAMES, rotation=20)
    ax.set_ylabel("Action frequency")
    ax.set_title("Action distribution")
    ax.legend(frameon=False)

    for ax_ in axes.ravel():
        ax_.spines["top"].set_visible(False)
        ax_.spines["right"].set_visible(False)
        ax_.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle(f"Behavioral analysis ({args.episodes} episodes per variant)", fontsize=13)
    fig.tight_layout()

    out = ASSETS_DIR / "behavior_distributions.png"
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
