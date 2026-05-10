"""
Shared helpers: vector environment construction and a callback that records
per-episode rewards for offline plotting (we don't depend on tensorboard for
the figures in the report).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from .config import EnvConfig, RewardVersion
from .env_wrapper import make_env


def make_vec_env(
    env_cfg: EnvConfig,
    reward_version: RewardVersion,
    n_envs: int,
    seed: int = 0,
    use_subproc: bool = False,
) -> VecEnv:
    """Create a vectorized env. DummyVecEnv is faster on laptops for small nets."""

    def _factory(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env = make_env(env_cfg, reward_version, seed=seed + rank)
            return env

        return _init

    fns = [_factory(i) for i in range(n_envs)]
    return SubprocVecEnv(fns) if use_subproc else DummyVecEnv(fns)


class EpisodeRewardLogger(BaseCallback):
    """Append (step, episode_reward, episode_length) rows to a CSV file.

    SB3's Monitor wrapper already tracks episode stats; this callback just
    flushes them to disk so plot_rewards.py can render publication-quality
    curves without touching tensorboard event files.
    """

    def __init__(self, csv_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.csv_path = csv_path
        self._initialized: bool = False
        self._episode_count: int = 0

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "timestep", "reward", "length"])
        self._initialized = True

    def _on_step(self) -> bool:
        # SB3 surfaces episode info in `infos` when an episode ends.
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is None:
                continue
            self._episode_count += 1
            with self.csv_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self._episode_count,
                        self.num_timesteps,
                        float(ep["r"]),
                        int(ep["l"]),
                    ]
                )
        return True


def smooth(values: list[float] | np.ndarray, window: int = 20) -> np.ndarray:
    """Moving-average smoothing for noisy reward curves."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or window <= 1:
        return arr
    kernel = np.ones(window) / window
    # 'same' mode keeps output length equal to input
    return np.convolve(arr, kernel, mode="same")
