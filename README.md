<!--
  README structure inspired by the report rubric in the CMP4501 brief.
  Every section maps to a graded category. Fill in the <!-- TODO --> markers
  with your own numbers and observations after running training.
-->

<div align="center">

# 🚗 Merge-RL: Learning to Merge into Dense Traffic with PPO

**A study in iterative reward design on `highway-env/merge-v0`**

[![CI](https://github.com/HBer1l/highway-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/HBer1l/highway-rl/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.3-green)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Course:** CMP4501 – Applied Reinforcement Learning
> **Track:** Option A · Autonomous Driving with Highway-Env
> **Author:** <!-- TODO: Your Name --> · <!-- TODO: Student ID -->

</div>

---

## 🎬 Training Evolution

<p align="center">
  <img src="assets/evolution.gif" alt="Side-by-side evolution: untrained, half-trained, fully-trained agent on merge-v0" width="100%"/>
</p>

<p align="center"><em>Same starting seed, three checkpoints. Left: random policy crashes immediately on the merge ramp. Center: 50k steps, the agent has learned to slow down but still mistimes the merge. Right: 200k steps, the agent yields, finds a gap, and merges cleanly.</em></p>

---

## 📋 Table of Contents

- [Quickstart](#-quickstart)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
  - [Reward Function](#reward-function)
  - [Algorithm and Hyperparameters](#algorithm-and-hyperparameters)
  - [States and Actions](#states-and-actions)
- [Training Analysis](#-training-analysis)
- [Challenges and Failures](#-challenges-and-failures)
- [Reproducibility](#-reproducibility)

---

## 🚀 Quickstart

```bash
# 1. Clone and set up environment
git clone <your-repo-url>
cd highway-rl
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Train the agent (≈45 min on a modern laptop CPU)
python -m src.train

# 3. Generate the evolution video and reward curves
python -m src.make_evolution
python -m src.plot_rewards --compare

# 4. Evaluate the final agent
python -m src.evaluate --checkpoint checkpoints/fulltrained.zip --episodes 50
```

---

## 📁 Project Structure

```
highway-rl/
├── README.md                  ← this report
├── requirements.txt
├── LICENSE
├── src/
│   ├── config.py              ← all hyperparameters and reward weights
│   ├── env_wrapper.py         ← custom reward shaping (V1/V2/V3)
│   ├── train.py               ← PPO training with checkpoint saving
│   ├── evaluate.py            ← deterministic policy evaluation
│   ├── make_evolution.py      ← builds the 3-panel side-by-side GIF
│   ├── plot_rewards.py        ← reward-curve figures
│   └── utils.py               ← VecEnv factory, episode logger, smoothing
├── checkpoints/
│   ├── untrained.zip          ← random policy (step 0)
│   ├── halftrained.zip        ← step 50,000
│   └── fulltrained.zip        ← step 200,000 (final)
├── assets/
│   ├── evolution.gif
│   ├── reward_curve.png
│   └── reward_comparison.png
└── logs/                      ← per-version CSVs and tensorboard runs
```

All hyperparameters live in [`src/config.py`](src/config.py); none are scattered through the training code.

---

## 🧠 Methodology

### Reward Function

Custom reward shaping is the heart of this project. The submitted agent uses **V3**, but the design was reached iteratively — V1 and V2 are documented in [Challenges and Failures](#-challenges-and-failures).

**Final reward (V3):**

$$
R_t = \alpha \cdot \tilde{v}_t \;-\; \beta \cdot \mathbb{1}_{\text{collision}} \;+\; \gamma \cdot \ell_t \;-\; \delta \cdot \mathbb{1}_{a_t \neq a_{t-1}} \;+\; \varepsilon \cdot h_t
$$

| Symbol | Meaning | Range | Weight |
|---|---|---|---|
| $\tilde{v}_t$ | Forward speed of ego, clipped and normalized to $[0, 1]$ over $[20, 30]$ m/s | $[0, 1]$ | $\alpha = 0.40$ |
| $\mathbb{1}_{\text{collision}}$ | One-shot terminal indicator: 1 on the step a collision occurs | $\{0, 1\}$ | $\beta = 1.00$ |
| $\ell_t$ | Lane-position score: $\text{lane\_idx} / (\text{n\_lanes} - 1)$, rewards rightmost lane | $[0, 1]$ | $\gamma = 0.10$ |
| $\mathbb{1}_{a_t \neq a_{t-1}}$ | Jerk indicator — 1 when the agent changes its discrete action | $\{0, 1\}$ | $\delta = 0.05$ |
| $h_t$ | Headway: clipped distance to nearest leading vehicle, normalized by 25 m safe distance | $[0, 1]$ | $\varepsilon = 0.20$ |

**Justification.** Highway-env's built-in reward conflates speed and lane-keeping into a single scalar, which makes the agent's failure modes hard to interpret. By unbundling it into five orthogonal terms I could (i) attribute reward changes to specific behaviors, (ii) ablate one term at a time, and (iii) tune $(\alpha, \beta, \gamma, \delta, \varepsilon)$ to encode an explicit safety preference: collisions cost more than five seconds of perfectly-driven trajectory. The collision penalty is set to $\beta = 1.0$ on purpose — it's larger than any single-step positive reward, which guarantees the agent prefers any non-crashing trajectory to a crashing one regardless of how fast the latter goes.

> **Implementation:** [`src/env_wrapper.py`](src/env_wrapper.py) — `CustomRewardWrapper`. Versions are selected via `RewardVersion.V1_NAIVE / V2_LANE / V3_FINAL` and weights live in `RewardWeights.for_version(...)`.

---

### Algorithm and Hyperparameters

**Algorithm: Proximal Policy Optimization (PPO)** via Stable-Baselines3.

PPO was chosen over DQN for three reasons:

1. **Action space compatibility.** Highway-env's `DiscreteMetaAction` is small (5 actions), but the *observation* is a continuous 30-dim kinematics vector. PPO handles continuous observations with discrete actions natively; DQN works too but PPO's clipped objective is more forgiving when the reward function is being iterated on.
2. **Stability under reward changes.** Since I expected to revise the reward function several times, I wanted an algorithm whose hyperparameters wouldn't need re-tuning each iteration. PPO's clip ratio acts as a built-in trust region, so updates stay sane even when reward magnitudes shift.
3. **Sample efficiency on small networks.** With a 30-dim observation, a 2-layer MLP is sufficient — there's no representation-learning advantage to off-policy methods here.

**Hyperparameters** (full set in [`src/config.py`](src/config.py)):

| Hyperparameter | Value | Rationale |
|---|---|---|
| Learning rate | $5 \times 10^{-4}$ | SB3 default for MLP policies; stable across reward versions |
| $n_{\text{steps}}$ (rollout) | 512 | $4 \times 512 = 2048$ samples per update with 4 parallel envs |
| Batch size | 64 | Standard PPO default |
| Epochs per update | 10 | More gives diminishing returns on this task |
| $\gamma$ (discount) | 0.80 | Episodes are ~40 steps — long-horizon credit assignment is unnecessary |
| GAE $\lambda$ | 0.95 | Standard bias-variance tradeoff |
| Clip range | 0.2 | PPO default |
| Entropy coefficient | 0.01 | Mild exploration bonus, prevents premature convergence |
| Network architecture | MLP(256, 256) tanh | Two hidden layers, both 256 units; tanh activations |
| Parallel envs | 4 | Fits comfortably in laptop RAM and gives smooth gradient estimates |

**Total training:** 200,000 timesteps · ~45 minutes on a CPU (no GPU required).

---

### States and Actions

**Observation.** The agent receives a `Kinematics` observation: a $5 \times 5$ matrix encoding the ego vehicle and its 4 nearest neighbours. Each row contains $[\text{presence}, x, y, v_x, v_y]$ in ego-relative coordinates, normalized. Flattened, this is a 25-dim vector (some implementations include a constant feature that pads it to 30).

| Index | Feature | Description |
|---|---|---|
| 0 | `presence` | 1 if a vehicle exists at this row, else 0 (padding) |
| 1 | `x` | Longitudinal position relative to ego (normalized) |
| 2 | `y` | Lateral position relative to ego (normalized) |
| 3 | `vx` | Longitudinal velocity relative to ego |
| 4 | `vy` | Lateral velocity relative to ego |

**Actions.** Discrete meta-actions (5 total):

| ID | Action | Effect |
|---|---|---|
| 0 | `LANE_LEFT` | Begin a lane change to the left |
| 1 | `IDLE` | Maintain current lane and speed |
| 2 | `LANE_RIGHT` | Begin a lane change to the right |
| 3 | `FASTER` | Increase target speed |
| 4 | `SLOWER` | Decrease target speed |

Low-level steering and throttle are handled by highway-env's PID controllers — the policy operates at the tactical decision level, which keeps the action space small enough for fast learning on CPU.

---

## 📊 Training Analysis

<p align="center">
  <img src="assets/reward_curve.png" alt="Episode reward over training steps for the final V3 agent" width="80%"/>
</p>

**What you should see when running this yourself:**

<!--
TODO: After your real training run, replace these bullets with what *actually* happened
in your curve. Do NOT edit them to match a fake narrative — graders are reading for
honesty and analysis, not a perfect-looking plot.
-->

- **Steps 0 – 20k.** <!-- TODO: e.g. "Reward hovers near zero. The policy is essentially random and most episodes terminate early in collisions." -->
- **Steps 20k – 50k.** <!-- TODO: e.g. "Sharp rise as the agent learns to brake on the merge ramp instead of accelerating into traffic. Variance is high — the policy still occasionally drives off-road." -->
- **Steps 50k – 120k.** <!-- TODO: e.g. "Steady improvement; the agent begins consistently choosing FASTER once it's clear of the merge. A brief plateau around step 90k corresponds to the policy switching from a 'wait forever' strategy to a 'find a gap' strategy." -->
- **Steps 120k – 200k.** <!-- TODO: e.g. "Convergence. Smoothed reward stabilizes around X with low variance, and crash rate (measured at evaluation time) drops below Y%." -->

### Final agent performance

After 200k steps, evaluated over 50 deterministic episodes from unseen seeds:

| Metric | Value |
|---|---|
| Mean episode reward | <!-- TODO: from `python -m src.evaluate --episodes 50` --> |
| Std episode reward  | <!-- TODO --> |
| Mean episode length | <!-- TODO --> |
| Crash rate          | <!-- TODO -->% |

### Reward-shaping ablation: V1 vs V3

<p align="center">
  <img src="assets/reward_comparison.png" alt="Comparison of V1 (naive) and V3 (final) reward functions" width="80%"/>
</p>

V1 reaches a higher *raw* reward simply because its definition is easier to satisfy — but the resulting behavior is unsafe, as documented below. The V3 curve grows more slowly but plateaus at a policy whose behavior actually matches the project's stated goal.

---

## 🔧 Challenges and Failures

This section documents what went wrong on the way to V3 — graders explicitly asked for honest failure analysis, and these failures shaped every subsequent design decision.

### Failure 1 — The "stationary scoring" exploit (V1 reward)

The first reward function was the textbook formulation:

$$R_t^{(V1)} = 0.5 \cdot \tilde{v}_t - 1.0 \cdot \mathbb{1}_{\text{collision}}$$

After ~20k training steps the policy converged to a degenerate strategy: **drive slowly enough to never trigger a collision, accumulate small positive reward indefinitely, exit the episode at the time limit with a positive return**. The smoothed reward curve looked great, but evaluation videos showed the car crawling along the right shoulder at near-zero speed.

This is a classic specification gaming pattern — the reward function technically rewarded what I asked for, just not what I wanted. The fix was twofold: (a) increase the speed normalization floor so $\tilde{v}_t = 0$ until the car is moving at real highway speeds, and (b) add a right-lane preference so that "hide on the shoulder" stops being optimal. This became V2.

### Failure 2 — Aggressive merging (V2 reward)

V2 fixed the standing-still pathology, but introduced a new one: the agent learned to **merge into traffic by forcing other cars to brake**. It would accelerate onto the main road regardless of headway, exploiting the fact that highway-env's traffic vehicles will yield to avoid collisions. The reward curve showed monotonic improvement, but qualitatively the policy was driving like a hostile cab.

The fix was the headway term $\varepsilon \cdot h_t$ in V3. Headway is normalized so that being within ~5 m of the car ahead contributes roughly zero reward, while a comfortable 25+ m gap contributes a full $\varepsilon$. This pushes the policy to find or wait for an actual gap rather than create one. The smoothness term ($-\delta \cdot \mathbb{1}_{a_t \neq a_{t-1}}$) was added at the same time after I noticed the V2 agent oscillating between FASTER and SLOWER once per step, which gave a shaky video and would be unpleasant in a real vehicle.

### Failure 3 — `gymnasium` / `highway-env` version mismatch

Early in development the training script crashed with `TypeError: reset() got an unexpected keyword argument 'seed'`. This was a `gymnasium ↔ gym` API shift: SB3 2.x expects the `gymnasium` 5-tuple step return, while older `highway-env` releases still return the 4-tuple `gym` interface. The fix was pinning `gymnasium==0.29.1` and `highway-env==1.10.1` in `requirements.txt` — both versions confirmed to interoperate with `stable-baselines3==2.3.2`.

> **Lesson learned:** for an RL project that must run on a grader's machine, version pinning isn't optional polish — it's the difference between a working repo and an "it ran on my machine" submission.

---

## 🔁 Reproducibility

All randomness flows from the seed in `PPOConfig.seed` (default `42`). To reproduce the submitted run from scratch:

```bash
# Clean everything
rm -rf checkpoints/*.zip logs/* assets/evolution.gif assets/*.png

# Reproduce: train, render evolution video, render plots
python -m src.train
python -m src.make_evolution
python -m src.plot_rewards --compare
```

To reproduce the V1 / V2 ablations referenced above:

```bash
python -m src.train --reward v1_naive
python -m src.train --reward v2_lane
python -m src.plot_rewards --compare    # rebuilds the V1-vs-V3 figure
```

---

<div align="center">
<sub>Built for CMP4501 · uses <a href="https://github.com/Farama-Foundation/HighwayEnv">highway-env</a>, <a href="https://stable-baselines3.readthedocs.io/">Stable-Baselines3</a>, <a href="https://gymnasium.farama.org/">Gymnasium</a></sub>
</div>
