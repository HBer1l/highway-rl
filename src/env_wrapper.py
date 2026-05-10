"""
Environment wrapper implementing custom reward shaping for merge-v0.

Three reward versions are supported (see config.RewardVersion):

  V1_NAIVE  – speed minus collision. Used to demonstrate reward hacking:
              the agent learns to crawl forward and then sit still to avoid
              ever crashing, scoring positive reward indefinitely.
  V2_LANE   – adds a right-lane bonus. Fixes the standing-still pathology
              but the agent still tailgates aggressively at merges.
  V3_FINAL  – adds headway (distance-to-front-vehicle) and jerk (action-
              change) terms. This is the version trained for submission.

The wrapper is intentionally thin — it delegates dynamics to highway-env and
only replaces the scalar reward. This makes ablations (turning off a single
term) trivial.
"""
from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import highway_env

# In gymnasium 1.x, importing highway_env no longer auto-registers envs.
# We must call this explicitly. Safe to call multiple times.
if hasattr(highway_env, "register_highway_envs"):
    highway_env.register_highway_envs()
import numpy as np
from gymnasium.core import ActType, ObsType

from .config import EnvConfig, RewardVersion, RewardWeights


class CustomRewardWrapper(gym.Wrapper):
    """Replaces highway-env's built-in reward with a versioned custom one.

    Reward terms (each in [0, 1] before weighting unless noted):
        v_norm     – normalized forward speed of ego
        collision  – 1.0 on the step a collision occurs, else 0
        right_lane – 1.0 if ego is in the rightmost lane, else 0
        jerk       – |a_t - a_{t-1}| / max_action_diff, in [0, 1]
        headway    – clip(dx_to_front / safe_distance, 0, 1)
    """

    def __init__(
        self,
        env: gym.Env,
        version: RewardVersion = RewardVersion.V3_FINAL,
    ) -> None:
        super().__init__(env)
        self.version: RewardVersion = version
        self.weights: RewardWeights = RewardWeights.for_version(version)
        self._last_action: int | None = None
        # Speed normalization range — merge-v0 vehicles cap around 30 m/s
        self._v_min: float = 20.0
        self._v_max: float = 30.0

    # ---- gym.Wrapper interface ------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self._last_action = None
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, _orig_reward, terminated, truncated, info = self.env.step(action)
        custom_reward = self._compute_reward(action, info)
        # Keep the original reward in info for comparison/debugging
        info["original_reward"] = float(_orig_reward)
        info["custom_reward"] = custom_reward
        info["reward_version"] = self.version.value
        self._last_action = int(action) if np.isscalar(action) else None
        return obs, custom_reward, terminated, truncated, info

    # ---- reward computation ---------------------------------------------
    def _compute_reward(self, action: ActType, info: dict[str, Any]) -> float:
        ego = self.env.unwrapped.vehicle  # type: ignore[attr-defined]

        # Forward speed term, normalized to [0, 1]
        v = float(np.linalg.norm(ego.velocity)) if ego is not None else 0.0
        v_norm = float(np.clip((v - self._v_min) / (self._v_max - self._v_min), 0.0, 1.0))

        # Collision term — terminal one-shot penalty
        crashed = bool(info.get("crashed", False)) or bool(getattr(ego, "crashed", False))
        collision = 1.0 if crashed else 0.0

        # Right-lane bonus (lane index increases left→right in highway-env)
        try:
            lane_idx = ego.lane_index[2] if ego is not None else 0
            road = self.env.unwrapped.road  # type: ignore[attr-defined]
            num_lanes = len(road.network.lanes_list())
            right_lane = float(lane_idx) / max(num_lanes - 1, 1)
        except Exception:  # pragma: no cover — defensive against env edge cases
            right_lane = 0.0

        # Jerk — penalize action changes (proxy for passenger comfort)
        if self._last_action is not None and np.isscalar(action):
            jerk = 1.0 if int(action) != self._last_action else 0.0
        else:
            jerk = 0.0

        # Headway — distance to nearest leading vehicle, clipped
        headway = self._headway(ego) if ego is not None else 1.0

        # TTC-based dense safety penalty (V3c only — zeta_ttc=0 for other versions)
        ttc_penalty = self._ttc_penalty(ego) if ego is not None else 0.0

        w = self.weights
        reward = (
            w.alpha_speed * v_norm
            - w.beta_collision * collision
            + w.gamma_right_lane * right_lane
            - w.delta_jerk * jerk
            + w.epsilon_headway * headway
            - w.zeta_ttc * ttc_penalty
        )
        return float(reward)

    @staticmethod
    def _ttc_penalty(
        ego: Any,
        ttc_safe: float = 4.0,
        ttc_critical: float = 1.0,
    ) -> float:
        """Dense time-to-collision penalty in [0, 1].

        Returns 0 when TTC > ttc_safe (no risk), 1 when TTC < ttc_critical
        (imminent collision), and linearly interpolated in between. This is
        the dense analog of a collision indicator: PPO can credit-assign over
        the ramp window rather than only at the terminal step.

        TTC is computed for the leading vehicle assuming both vehicles
        maintain current longitudinal velocities. Lateral closing speed and
        non-leading vehicles are not modeled — a simplification consistent
        with the kinematics observation the agent receives.
        """
        try:
            road = ego.road
            front, _ = road.neighbour_vehicles(ego)
            if front is None:
                return 0.0
            dx = float(front.position[0] - ego.position[0])
            if dx <= 0.0:
                return 1.0  # already colliding/passed
            dv = float(ego.velocity[0] - front.velocity[0])
            if dv <= 0.0:
                return 0.0  # ego is slower than lead — no closing
            ttc = dx / dv
            if ttc >= ttc_safe:
                return 0.0
            if ttc <= ttc_critical:
                return 1.0
            # linear ramp between critical and safe
            return (ttc_safe - ttc) / (ttc_safe - ttc_critical)
        except Exception:
            return 0.0

    @staticmethod
    def _headway(ego: Any, safe_distance: float = 25.0) -> float:
        """Distance to closest vehicle in front of ego, normalized to [0, 1].

        Returns 1.0 when there is no leading vehicle within range (i.e. open
        road). Returns ~0 when tailgating.
        """
        try:
            road = ego.road
            front, _ = road.neighbour_vehicles(ego)
            if front is None:
                return 1.0
            dx = float(front.position[0] - ego.position[0])
            if dx <= 0:
                return 0.0
            return float(np.clip(dx / safe_distance, 0.0, 1.0))
        except Exception:
            return 1.0


def make_env(
    env_cfg: EnvConfig,
    reward_version: RewardVersion,
    render_mode: str | None = None,
    seed: int | None = None,
) -> gym.Env:
    """Factory for a single wrapped merge-v0 environment.

    Use this everywhere — train.py, evaluate.py, make_evolution.py — so that
    the reward function and configuration stay consistent across scripts.
    """
    env = gym.make(env_cfg.env_id, render_mode=render_mode, config=env_cfg.to_dict())
    env = CustomRewardWrapper(env, version=reward_version)
    if seed is not None:
        env.reset(seed=seed)
    return env
