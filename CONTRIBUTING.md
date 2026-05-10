# Contributing

This is an academic course project (CMP4501 — Applied Reinforcement Learning), so external pull requests are not actively maintained. The repository is preserved as-is for portfolio and reproducibility purposes.

If you're studying this project for your own learning, you're welcome to:

- **Open issues** with questions about the methodology, results, or code
- **Fork** the repository and adapt it for your own experiments
- **Cite** this work in academic contexts (see [`CITATION.cff`](CITATION.cff))

## Reproducing the results

The full experimental pipeline is documented in [`README.md`](README.md#-reproducibility). All randomness flows through the `--seed` flag; identical seeds on the same hardware produce identical training trajectories (CPU is deterministic; GPU is not, by design of `cuDNN`).

Trained checkpoints from the original study are committed to `checkpoints/` for reproducibility without retraining (~5 minutes to re-evaluate vs ~22 minutes per training seed).

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you modify `src/env_wrapper.py` or `src/config.py`, smoke-test before training:

```bash
python -c "
from src.env_wrapper import make_env
from src.config import CONFIG, RewardVersion
env = make_env(CONFIG.env, RewardVersion.V3_TTC)
obs, _ = env.reset(seed=0)
print('OK', obs.shape, env.step(env.action_space.sample())[1])
"
```

## Code style

- Python 3.10+
- PEP8, type hints throughout
- Hyperparameters live in `src/config.py`, never inline
- All new reward variants must be added to the `RewardVersion` enum and `RewardWeights.for_version()` so they integrate with the existing training and evaluation infrastructure

## Reporting issues

When reporting bugs or unexpected results, please include:

- Operating system and Python version
- `pip freeze` output
- Random seed used
- Relevant log file from `logs/`
- Hardware (CPU model — GPU is not used)
