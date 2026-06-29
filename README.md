# FastDSAC: Enhancing Policy Plasticity via Constrained Exploration for Scalable Humanoid Locomotion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FastDSAC is an off-policy reinforcement learning framework for scalable humanoid locomotion. The repository contains both the DSAC baseline and the proposed FastDSAC method.

FastDSAC retains the stochastic actor and distributional twin critics of DSAC while applying mean-centered truncation when generating target actions. The DSAC implementation uses the standard tanh-squashed Gaussian policy without the truncation mechanism.

## Features

- FastDSAC and DSAC implementations in the same codebase
- Distributional twin critics
- Support for HumanoidBench and MuJoCo Playground
- Observation and reward normalization
- Automatic mixed precision and optional `torch.compile`
- Checkpoint saving, evaluation, logging, and rollout rendering
- GPU replay buffer for efficient massively parallel training

## Repository Structure

```text
FastDSAC/
├── data/
├── fast_dsac/
│   ├── environments/
│   │   ├── humanoid_bench_env.py
│   │   └── mujoco_playground_env.py
│   ├── __init__.py
│   ├── dsact_net.py
│   ├── dsact.py
│   ├── fast_dsac_utils.py
│   ├── fastdsac_net.py
│   ├── fastdsac.py
│   ├── hyperparams.py
│   ├── train_multigpu.py
│   └── training_notebook.ipynb
├── log/
├── models/
├── requirements/
├── README.md
├── setup.py
└── sim2real.md
```

## Installation

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd FastDSAC
```

Install the package in editable mode:

```bash
pip install -e .
```

### HumanoidBench Environment

```bash
conda create -n fastdsac_hb -y python=3.10
conda activate fastdsac_hb

pip install --editable git+https://github.com/carlosferrazza/humanoid-bench.git#egg=humanoid-bench
pip install -r requirements/requirements.txt
pip install -e .
```

### MuJoCo Playground Environment

```bash
conda create -n fastdsac_playground -y python=3.10
conda activate fastdsac_playground

pip install -r requirements/requirements_playground.txt
pip install -e .
```

## Running Experiments

Run all commands from the repository root directory.

### FastDSAC on HumanoidBench

```bash
python fast_dsac/fastdsac.py \
    --env_name h1hand-hurdle-v0 \
    --exp_name FastDSAC \
    --render_interval 5000 \
    --seed 1
```

### DSAC on HumanoidBench

```bash
python fast_dsac/dsact.py \
    --env_name h1hand-hurdle-v0 \
    --exp_name DSAC \
    --render_interval 5000 \
    --seed 1
```

### FastDSAC on MuJoCo Playground

```bash
python fast_dsac/fastdsac.py \
    --env_name T1JoystickFlatTerrain \
    --exp_name FastDSAC \
    --render_interval 5000 \
    --seed 1
```

### DSAC on MuJoCo Playground

```bash
python fast_dsac/dsact.py \
    --env_name T1JoystickFlatTerrain \
    --exp_name DSAC \
    --render_interval 5000 \
    --seed 1
```

The available arguments and default hyperparameters are defined in `fast_dsac/hyperparams.py`.

For boolean arguments, a flag can be disabled by adding the `no_` prefix when supported by the argument parser. For example:

```bash
--no_use_cdq
```

## Task-Specific Hyperparameters

The following MuJoCo Playground overrides are applied only to FastDSAC. In the implementation, `policy_noise` denotes the mean-centered truncation radius \(c\).

| Environment | `bias` | Truncation radius `policy_noise` |
|---|---:|---:|
| `G1JoystickFlatTerrain` | `1e-3` | `1e-3` |
| `G1JoystickRoughTerrain` | `1e-3` | `1e-2` |
| `T1JoystickFlatTerrain` | `0.1` | `1e-3` |
| `T1JoystickRoughTerrain` | `0.1` | `1e-2` |

These G1/T1 overrides are not applied to the DSAC baseline.

For all HumanoidBench environments, both FastDSAC and DSAC use:

```text
log_std_init = -2.0
```

The supported HumanoidBench experiments are:

- `h1hand-run-v0`
- `h1hand-walk-v0`
- `h1hand-stand-v0`
- `h1hand-pole-v0`
- `h1hand-slide-v0`
- `h1hand-hurdle-v0`

## Parameter Entry Points

The two training scripts must identify the algorithm when loading the task configuration.

In `fastdsac.py`:

```python
args = get_args("fastdsac")
```

In `dsact.py`:

```python
args = get_args("dsac")
```

This separation ensures that the FastDSAC-specific `bias` and truncation-radius defaults do not alter the DSAC baseline.

## Package Imports

The package exposes separate names for the DSAC and FastDSAC networks:

```python
from fast_dsac import (
    DSACActor,
    DSACCritic,
    FastDSACActor,
    FastDSACCritic,
)
```

Direct module imports are also supported:

```python
from fast_dsac.dsact_net import Actor as DSACActor
from fast_dsac.dsact_net import Critic as DSACCritic

from fast_dsac.fastdsac_net import Actor as FastDSACActor
from fast_dsac.fastdsac_net import Critic as FastDSACCritic
```

## Output Directories

Training outputs are stored in the following directories:

- `log/`: experiment logs and rendered rollouts
- `models/`: saved checkpoints
- `data/`: experiment data used for analysis or plotting

## License

This project is released under the MIT License.

## Acknowledgements

This repository was developed from the FastTD3 codebase and builds on the LeanRL framework. It also uses HumanoidBench and MuJoCo Playground for evaluation.

## Citation

```bibtex
@article{lu2026fastdsac,
  title={FastDSAC: Enhancing Policy Plasticity via Constrained Exploration for Scalable Humanoid Locomotion},
  author={Guanchen Lu and Yajuan Dun and Yi Zhou and Letian Tao and Jingliang Duan and Jie Li and Guofa Li},
  year={2026}
}
```
