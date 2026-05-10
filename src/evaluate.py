"""
Evaluate a trained checkpoint and (optionally) record a video.

Usage:
    python -m src.evaluate --checkpoint checkpoints/fulltrained.zip
    python -m src.evaluate --checkpoint checkpoints/fulltrained.zip --record videos/final.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO

from .config import CONFIG, Config, RewardVersion
from .env_wrapper import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--reward", type=str, default=RewardVersion.V3_FINAL.value)
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="If set, save an MP4 of the first episode to this path.",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def evaluate(
    cfg: Config,
    checkpoint: Path,
    reward_version: RewardVersion,
    n_episodes: int,
    record_path: Path | None = None,
    seed: int = 123,
) -> dict[str, float]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    render_mode = "rgb_array" if record_path is not None else None
    env = make_env(cfg.env, reward_version, render_mode=render_mode, seed=seed)
    model = PPO.load(str(checkpoint))

    returns: list[float] = []
    lengths: list[int] = []
    crash_count: int = 0
    frames: list[np.ndarray] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0
        while not done:
            if record_path is not None and ep == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            done = bool(terminated or truncated)
            if done and info.get("crashed", False):
                crash_count += 1
        returns.append(ep_return)
        lengths.append(ep_len)

    env.close()

    if record_path is not None and frames:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(record_path), frames, fps=15)
        print(f"[eval] saved video → {record_path}")

    stats = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "crash_rate": crash_count / n_episodes,
    }
    print(
        f"[eval] {checkpoint.name}: "
        f"return = {stats['mean_return']:.2f} ± {stats['std_return']:.2f} | "
        f"length = {stats['mean_length']:.1f} | "
        f"crash rate = {stats['crash_rate']:.1%}"
    )
    return stats


def main() -> None:
    args = parse_args()
    evaluate(
        cfg=CONFIG,
        checkpoint=args.checkpoint,
        reward_version=RewardVersion(args.reward),
        n_episodes=args.episodes,
        record_path=args.record,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
