"""
Train a PPO agent on merge-v0 with the custom reward wrapper.

The script saves three named checkpoints required by the evolution video:
    - checkpoints/untrained.zip   (random policy, before any updates)
    - checkpoints/halftrained.zip (saved at TrainingConfig.checkpoint_half_steps)
    - checkpoints/fulltrained.zip (saved at TrainingConfig.checkpoint_full_steps)

It also writes per-episode rewards to logs/<reward_version>/rewards.csv so
plot_rewards.py can produce the figures embedded in the README.

Usage:
    python -m src.train                          # train final V3 reward
    python -m src.train --reward v1_naive        # reproduce the naive failure
    python -m src.train --reward v2_lane         # reproduce the V2 ablation
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from .config import (
    CHECKPOINT_DIR,
    CONFIG,
    LOG_DIR,
    Config,
    RewardVersion,
)
from .env_wrapper import make_env
from .utils import EpisodeRewardLogger, make_vec_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on highway-env merge-v0.")
    parser.add_argument(
        "--reward",
        type=str,
        default=RewardVersion.V3_FINAL.value,
        choices=[v.value for v in RewardVersion],
        help="Which reward function version to train with.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override total training timesteps (default: TrainingConfig.total_timesteps).",
    )
    parser.add_argument(
        "--save-as-production",
        action="store_true",
        help="Save checkpoints as untrained.zip / halftrained.zip / fulltrained.zip "
             "instead of <version>_final.zip. Use for the canonical 'production' agent "
             "that the evolution video will use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the PPO seed (for multi-seed runs).",
    )
    return parser.parse_args()


def train(
    cfg: Config,
    reward_version: RewardVersion,
    total_steps: int | None = None,
    save_as_production: bool = False,
    seed_override: int | None = None,
) -> None:
    total = total_steps or cfg.training.total_timesteps
    print(f"[train] reward={reward_version.value}  total_timesteps={total:,}")

    # ---- output directories ------------------------------------------------
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_subdir = LOG_DIR / reward_version.value
    log_subdir.mkdir(parents=True, exist_ok=True)

    # ---- build envs --------------------------------------------------------
    seed = seed_override if seed_override is not None else cfg.ppo.seed
    vec_env = make_vec_env(
        env_cfg=cfg.env,
        reward_version=reward_version,
        n_envs=cfg.env.n_envs,
        seed=seed,
    )
    vec_env = VecMonitor(vec_env, filename=str(log_subdir / "monitor"))

    # ---- model -------------------------------------------------------------
    ppo_kwargs = cfg.ppo.to_kwargs()
    ppo_kwargs["seed"] = seed   # override PPO's RNG, not just the env's
    model = PPO(
        env=vec_env,
        tensorboard_log=str(log_subdir),
        **ppo_kwargs,
    )

    # ---- save the untrained checkpoint (only for the production agent) ----
    if save_as_production:
        untrained_path = CHECKPOINT_DIR / "untrained.zip"
        model.save(untrained_path)
        print(f"[train] saved {untrained_path}")

    # ---- training in two phases so we can save halftrained ----------------
    callback = EpisodeRewardLogger(csv_path=log_subdir / "rewards.csv")

    half = cfg.training.checkpoint_half_steps
    if save_as_production and half < total:
        model.learn(total_timesteps=half, callback=callback, progress_bar=True)
        half_path = CHECKPOINT_DIR / "halftrained.zip"
        model.save(half_path)
        print(f"[train] saved {half_path}")
        # Continue training to total
        model.learn(
            total_timesteps=total - half,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=True,
        )
    else:
        model.learn(total_timesteps=total, callback=callback, progress_bar=True)

    # ---- save final --------------------------------------------------------
    if save_as_production:
        final_path = CHECKPOINT_DIR / "fulltrained.zip"
    else:
        suffix = f"_seed{seed_override}" if seed_override is not None else ""
        final_path = CHECKPOINT_DIR / f"{reward_version.value}{suffix}_final.zip"
    model.save(final_path)
    print(f"[train] saved {final_path}")

    vec_env.close()


def main() -> None:
    args = parse_args()
    reward_version = RewardVersion(args.reward)
    train(
        CONFIG,
        reward_version,
        total_steps=args.total_steps,
        save_as_production=args.save_as_production,
        seed_override=args.seed,
    )


if __name__ == "__main__":
    main()
