# DeepCubeAI

This repository contains code for the paper [Learning Discrete World Models for Heuristic Search](https://rlj.cs.umass.edu/2024/papers/Paper225.html).

| ![Rubik's Cube solving animation](https://raw.githubusercontent.com/misaghsoltani/DeepCubeAI/master/images/dcai_rubiks_cube.gif) | ![Sokoban puzzle solving animation](https://raw.githubusercontent.com/misaghsoltani/DeepCubeAI/master/images/dcai_sokoban.gif) | ![Ice Slider puzzle solving animation](https://raw.githubusercontent.com/misaghsoltani/DeepCubeAI/master/images/dcai_iceslider.gif) | ![Digit Jump puzzle solving animation](https://raw.githubusercontent.com/misaghsoltani/DeepCubeAI/master/images/dcai_digitjump.gif) |
| :------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: |

## About DeepCubeAI

DeepCubeAI is an algorithm that learns a discrete world model and employs Deep Reinforcement Learning methods to learn a heuristic function that generalizes over start and goal states. We then integrate the learned model and the learned heuristic function with heuristic search, such as Q* search, to solve sequential decision making problems. For more details, please refer to the [paper](https://rlj.cs.umass.edu/2024/papers/Paper225.html).

## Quick links

- Key contributions: [Key Contributions](#key-contributions)
- Main results: [Main Results](#main-results)
- Quick start: [Quick start](#quick-start)
- Install: [docs/installation.md](https://github.com/misaghsoltani/DeepCubeAI/blob/main/docs/installation.md)
- CLI reference: [docs/cli.md](https://github.com/misaghsoltani/DeepCubeAI/blob/main/docs/cli.md)
- Stage-by-stage usage (all flags and paths): [docs/usage.md](https://github.com/misaghsoltani/DeepCubeAI/blob/main/docs/usage.md)
- Reproduce the paper results: [docs/reproduce.md](https://github.com/misaghsoltani/DeepCubeAI/blob/main/docs/reproduce.md)
- SLURM and Distributed training: [docs/qlearning_distributed.md](https://github.com/misaghsoltani/DeepCubeAI/blob/main/docs/qlearning_distributed.md)
- Environments and integration: [docs/environments.md](https://github.com/misaghsoltani/DeepCubeAI/blob/main/docs/environments.md)
- Python usage (API snippets): [docs/python_api.md](https://github.com/misaghsoltani/DeepCubeAI/blob/main/docs/python_api.md)
- Citing the paper: [Citation](#citation)
- Contact: [Contact](#contact)

## Key Contributions

### Overview

DeepCubeAI is comprised of three key components:

1. **Discrete World Model**

   - Learns a world model that represents states in a discrete latent space.
   - This approach tackles two challenges: model degradation and state re-identification.
   - Prediction errors less than 0.5 are corrected by rounding.
   - Re-identifies states by comparing two binary vectors.

   <br>

   | ![DeepCubeAI discrete world model](https://raw.githubusercontent.com/misaghsoltani/DeepCubeAI/master/images/dcai_discrete_world_model.png) |
   | :----------------------------------------------------------------------------------------------------------------------------------------: |

2. **Generalizable Heuristic Function**

   - Utilizes Deep Q-Network (DQN) and hindsight experience replay (HER) to learn a heuristic function that generalizes over start and goal states.

3. **Optimized Search**

   - Integrates the learned model and the learned heuristic function with heuristic search to solve problems. It uses [Q* search](https://prl-theworkshop.github.io/prl2024-icaps/papers/9.pdf), a variant of A* search optimized for DQNs, which enables faster and more memory-efficient planning.
‌

### Main Results

- Accurate reconstruction of ground truth images after thousands of timesteps.
- Achieved 100% success on Rubik's Cube (canonical goal), Sokoban, IceSlider, and DigitJump.
- 99.9% success on Rubik's Cube with reversed start/goal states.
- Demonstrated significant improvement in solving complex planning problems and generalizing to unseen goals.

## Quick start

DeepCubeAI provides a Python package and CLI. You can install it from PyPI or build it from source. The package supports Python 3.10-3.12.

> [!NOTE]
>
> You can find detailed installation instructions, including using Conda for environment management, in the [installation guide](https://github.com/misaghsoltani/DeepCubeAI/blob/main/docs/installation.md).

### Install `deepcubeai` Package from PyPI with [uv](https://docs.astral.sh/uv/) (Recommended if Running as a Package)

`deepcubeai` is available on PyPI and you can use the following commands to install it.

  1. **Install `uv`** from the official website: [Install uv](https://docs.astral.sh/uv/getting-started/installation/).
  2. Create and activate a virtual environment:

     ```bash
     # create a .venv in the current folder
     uv venv

     # macOS & Linux
     source .venv/bin/activate

     # Windows (PowerShell)
     .venv\Scripts\activate
     ```

     If you have multiple Python versions, ensure you use a supported one (3.10-3.12), e.g.:

     ```bash
     uv venv --python 3.12
     ```

  3. Install the package (using [uv’s pip interface](https://docs.astral.sh/uv/pip/)):

     ```bash
     uv pip install deepcubeai
     ```

### Install from Source with Pixi (Recommended if Working from Source)

[Pixi](https://pixi.sh/) is a package management tool that provides fast, reproducible environments with support for Conda and PyPI dependencies. The `pixi.toml` and `pixi.lock` files define reproducible environments with exact dependency versions.

1. **Install Pixi**: Follow the [official installation guide](https://pixi.sh/latest/installation/)
2. **Clone repository**:

   ```bash
   git clone https://github.com/misaghsoltani/DeepCubeAI.git
   cd DeepCubeAI
   ```

3. **Enter the default environment** (first run performs dependency resolution):

   ```bash
   pixi shell  # or: pixi shell -e default

   # or

   pixi install -e default # non-interactive solve only
   ```

### Running DeepCubeAI

For running the CLI use the following command to see the available options:

   ```bash
   # If already entered the environment with Pixi:
   deepcubeai --help  # or -h

   # or

   # Without entering the environment:
   pixi run deepcubeai --help  # or -h
   ```

Or use it as a Python package:

```python
import deepcubeai

print(deepcubeai.__version__)
```

## License

MIT License - see [LICENSE](LICENSE).

## Citation

If you use DeepCubeAI in your research, please cite:

```bibtex
@article{agostinelli2025learning,
    title={Learning Discrete World Models for Heuristic Search},
    author={Agostinelli, Forest and Soltani, Misagh},
    journal={Reinforcement Learning Journal},
    volume={4},
    pages={1781--1792},
    year={2025}
}
```

## Contact

If you have any questions or issues, please contact Misagh Soltani ([msoltani@email.sc.edu](mailto:msoltani@email.sc.edu))
