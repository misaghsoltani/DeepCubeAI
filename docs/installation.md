# Installation

DeepCubeAI provides a Python package and CLI (`deepcubeai` / `python -m deepcubeai`).

* Supported Python: 3.10 - 3.12

## 1. Install `deepcubeai` Package from PyPI

`deepcubeai` is available on PyPI and supports **Python 3.10-3.12**.

### Option A - Install with [uv](https://docs.astral.sh/uv/) (Recommended If Using the Package)

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

### Option B - Install with [pip](https://pip.pypa.io/en/stable/)

  1. Install pip from the official website: [Install pip](https://pip.pypa.io/en/stable/installation/).
  2. (Recommended) Create and activate a virtual environment:
     Create a .venv in the current folder:

     ```bash
     python -m venv .venv
     ```

     Activate the virtual environment:

     ```bash
     # macOS & Linux
     source .venv/bin/activate

     # Windows (PowerShell)
     .venv\Scripts\activate
     ```

     Install the package:

     ```bash
     pip install deepcubeai
     ```

     See: [pip install command](https://pip.pypa.io/en/stable/cli/pip_install/).

## 2. Source install with Pixi (Recommended when Installing from Source)

[Pixi](https://pixi.sh/) is a package management tool that provides fast, reproducible environments with support for Conda and PyPI dependencies. The `pixi.toml` and `pixi.lock` files define reproducible environments with exact dependency versions.

### Installation steps

1. **Install Pixi**: Follow the [official installation guide](https://pixi.sh/latest/installation/)
2. **Clone repository**:

   ```bash
   git clone https://github.com/misaghsoltani/DeepCubeAI.git
   cd DeepCubeAI
   ```

3. **Enter the default environment** (first run performs dependency resolution):

   ```bash
   pixi shell          # or: pixi shell -e default
   # non-interactive solve only:
   pixi install -e default
   ```

4. **Verify installation**:

   ```bash
   deepcubeai --help
   ```

### 2.1 Available environments

Pixi environments are defined in the `[environments]` section of `pixi.toml`. Each environment includes different feature sets for specific use cases:

| Name     | Description                                         |
| -------- | --------------------------------------------------- |
| default  | Core runtime dependencies                           |
| dev      | Development tools: ruff, mypy, pyright, shellcheck  |
| cuda     | CUDA 12.x runtime/toolkit & cuDNN (Linux/Windows)   |
| build    | Build tools (hatch)                                 |
| all      | Complete development environment (cuda, dev, build) |
| glibc217 | All features with glibc 2.17 compatibility          |

**Activate an environment**:

```bash
pixi shell -e dev
pixi shell -e cuda
pixi shell -e all
```

All environments share the same solve-group (`default`) for consistent dependency resolution. See [Pixi's environment documentation](https://pixi.sh/latest/features/environment/) for more details.

### 2.2 Development tasks

The `dev` feature includes predefined [tasks](https://pixi.sh/latest/workspace/advanced_tasks/) for code quality and type checking. Run these commands inside an environment that includes the `dev` feature:

```bash
pixi run lint       # ruff check --fix
pixi run ulint      # ruff check --fix --unsafe-fixes
pixi run format     # ruff format
pixi run fix        # ruff check --fix --unsafe-fixes; ruff format
pixi run mypy       # mypy type check on 'deepcubeai/'
pixi run pyright    # pyright type check on 'deepcubeai/'
```

### 2.3 Running the project

```bash
pixi run deepcubeai -- -h   # pass extra args after '--'
```

### 2.4 Building distributions

Use the build environment for creating distribution packages:

```bash
pixi shell -e build
pixi run build   # hatch build -t wheel -t sdist
```

Alternatively, invoke hatch directly if available in your PATH:

```bash
hatch build -t wheel -t sdist
```

Distribution artifacts will be created in the `dist/` directory.

## 3. PyPI (binary / sdist) install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install deepcubeai
deepcubeai --help
```

## 4. Source install with uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager and project manager that can replace pip, virtualenv, and other tools. It provides fast dependency resolution and environment management.

### Setup steps

1. **Install UV**: Follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/)
2. **Clone repository**:

   ```bash
   git clone https://github.com/misaghsoltani/DeepCubeAI.git
   cd DeepCubeAI
   ```

3. **Install the project**:

   ```bash
   # uv will automatically create a virtual environment and install dependencies
   uv sync
   
   # Activate the environment
   source .venv/bin/activate  # Linux/macOS
   # Or on Windows: .venv\Scripts\activate
   ```

4. **Verify installation**:

   ```bash
   uv run deepcubeai --help
   # or after activation (source .venv/bin/activate)
   deepcubeai --help
   ```

See [uv's documentation](https://docs.astral.sh/uv/) for more usage and features.

## 5. Conda

Use the provided environment files:

```bash
# Default
conda env create -f environment.yml -n deepcubeai

# Development (adds lint/type tools)
conda env create -f environment_dev.yml -n deepcubeai_dev
```

This will install the required packages.

Activate the environment:

```bash
conda activate deepcubeai   # or: conda activate deepcubeai_dev
```

Or you can install from source within a Conda environment:

```bash
# Editable source install
uv pip install -e . # Using uv
# or
pip install -e . # Using pip
```

## Verify Installation

**Check installation**:

For package:

  ```bash
  python -c "import deepcubeai; print(deepcubeai.__version__)"
  ```

For CLI:

  ```bash
  deepcubeai --help
  ```

**Quick smoke test**:

```bash
deepcubeai gen_offline --env cube3 --num_offline_steps 5 --num_train_eps 9 --num_val_eps 1
```

## Dependencies

**Core Python dependencies** (see `pixi.toml` or `pyproject.toml`): `torch`, `numpy`, `matplotlib`, `networkx`, `opencv-python`, `tensorboard`, `gymnasium`.
