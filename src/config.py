"""
Central configuration for the merge-v0 PPO experiment.

All hyperparameters, environment settings, and reward weights live here.
Keeping them separate from training logic makes experiments reproducible
and lets us version reward functions cleanly (see RewardVersion enum).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
LOG_DIR: Path = PROJECT_ROOT / "logs"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
VIDEO_DIR: Path = PROJECT_ROOT / "videos"


# ----------------------------------------------------------------------------
# Reward function versions
# ----------------------------------------------------------------------------
class RewardVersion(str, Enum):
    """Reward shaping versions, used to demonstrate iterative design.

    V1 is intentionally naive to expose reward-hacking behaviors; V3 is the
    final tuned version used for the submitted agent. The README documents
    failure modes observed on V1 and V2 and how they motivated each change.
    """

    V1_NAIVE = "v1_naive"          # speed - collision: agent learns to crash slowly
    V2_LANE = "v2_lane"            # + right-lane preference: still too aggressive
    V3_FINAL = "v3_final"          # + headway and smoothness: V3a, lenient collision
    V3_STRICT = "v3_strict"        # V3b: same shape, 5x stronger collision penalty
    V3_TTC = "v3_ttc"              # V3c: dense TTC-based safety signal (production candidate)


@dataclass(frozen=True)
class RewardWeights:
    """Coefficients for each reward term.

    The agent maximizes:
        R_t = alpha * v_norm
            - beta  * collision
            + gamma * right_lane
            - delta * jerk
            + eps   * headway
    where each term is in [-1, 1] or [0, 1] before weighting.
    """

    alpha_speed: float = 0.4         # forward velocity reward (normalized)
    beta_collision: float = 1.0      # one-shot terminal penalty
    gamma_right_lane: float = 0.1    # encourages right-lane discipline
    delta_jerk: float = 0.05         # penalizes erratic action changes
    epsilon_headway: float = 0.2     # rewards safe following distance
    zeta_ttc: float = 0.0            # V3c only: weight on dense TTC penalty

    @classmethod
    def for_version(cls, version: RewardVersion) -> "RewardWeights":
        """Return the weight set associated with a given reward version."""
        if version is RewardVersion.V1_NAIVE:
            return cls(
                alpha_speed=0.5,
                beta_collision=1.0,
                gamma_right_lane=0.0,
                delta_jerk=0.0,
                epsilon_headway=0.0,
            )
        if version is RewardVersion.V2_LANE:
            return cls(
                alpha_speed=0.4,
                beta_collision=1.0,
                gamma_right_lane=0.1,
                delta_jerk=0.0,
                epsilon_headway=0.0,
            )
        if version is RewardVersion.V3_STRICT:
            return cls(
                alpha_speed=0.4,
                beta_collision=5.0,
                gamma_right_lane=0.1,
                delta_jerk=0.05,
                epsilon_headway=0.2,
            )
        if version is RewardVersion.V3_TTC:
            # V3c: TTC-based dense safety + reduced collision (TTC handles most of it)
            return cls(
                alpha_speed=0.4,
                beta_collision=2.0,
                gamma_right_lane=0.1,
                delta_jerk=0.05,
                epsilon_headway=0.0,   # subsumed by TTC
                zeta_ttc=0.5,
            )
        # V3_FINAL — original full shaping (kept for ablation comparison)
        return cls()


# ----------------------------------------------------------------------------
# Environment configuration (highway-env 'merge-v0')
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvConfig:
    """Configuration passed to gymnasium.make for highway-env merge-v0.

    The kinematics observation gives the agent normalized [presence, x, y, vx,
    vy] features for itself and the 5 closest vehicles — a compact 30-dim
    vector that PPO can learn from quickly on CPU.
    """

    env_id: str = "merge-v0"
    n_envs: int = 4                          # parallel envs for PPO rollout
    simulation_frequency: int = 15           # Hz
    policy_frequency: int = 5                # agent acts at this rate
    duration: int = 40                       # episode length in env steps
    vehicles_count: int = 10
    vehicles_density: float = 1.0
    collision_reward: float = -1.0           # internal — overridden by wrapper
    high_speed_reward: float = 0.4
    right_lane_reward: float = 0.1
    lane_change_reward: float = 0.0

    def to_dict(self) -> dict:
        """Highway-env config dict format."""
        return {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "normalize": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "simulation_frequency": self.simulation_frequency,
            "policy_frequency": self.policy_frequency,
            "duration": self.duration,
            "vehicles_count": self.vehicles_count,
            "vehicles_density": self.vehicles_density,
            "collision_reward": self.collision_reward,
            "high_speed_reward": self.high_speed_reward,
            "right_lane_reward": self.right_lane_reward,
            "lane_change_reward": self.lane_change_reward,
        }


# ----------------------------------------------------------------------------
# PPO hyperparameters
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PPOConfig:
    """Stable-Baselines3 PPO hyperparameters tuned for merge-v0 on CPU.

    Defaults chosen for ~30-60 min training on a modern laptop CPU. The
    network is intentionally small (two 256-unit MLP layers) — the kinematics
    observation is low-dimensional and bigger nets just slow training without
    improving final performance.
    """

    policy: str = "MlpPolicy"
    learning_rate: float = 5e-4
    n_steps: int = 512                       # rollout length per env
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.8                       # short horizon — episodes are short
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01                   # mild exploration bonus
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (256, 256)
    seed: int = 42

    def to_kwargs(self) -> dict:
        return {
            "policy": self.policy,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "policy_kwargs": {"net_arch": list(self.net_arch)},
            "seed": self.seed,
            "verbose": 1,
            "device": "cpu",
        }


# ----------------------------------------------------------------------------
# Training schedule
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingConfig:
    """Total timesteps and checkpoint cadence for the evolution video.

    We deliberately save at three named milestones so make_evolution.py can
    render the untrained / half-trained / fully-trained side-by-side panels.
    """

    total_timesteps: int = 200_000
    checkpoint_untrained_steps: int = 0           # save before any training
    checkpoint_half_steps: int = 50_000           # ~25% of total
    checkpoint_full_steps: int = 200_000          # final
    eval_freq: int = 10_000
    n_eval_episodes: int = 10


# Convenience: a single object that callers can import
@dataclass(frozen=True)
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward_version: RewardVersion = RewardVersion.V3_FINAL


CONFIG = Config()
