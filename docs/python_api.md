# Python API Examples

This project exposes programmatic (Python) entrypoints for the main pipelines via small typed dataclasses and `run_*` functions. The examples below show how to import a config dataclass, construct it, then call the corresponding `run_*` function.

> [!NOTE]
>
> Each module mirrors the CLI flags. Config dataclasses expose the most common fields and provide a `from_json()` helper if you prefer JSON-based configs.

> [!NOTE]
>
> Some config fields named `env_model` or `env_dir` expect a model directory (for example `deepcubeai/saved_env_models/<name>`), not a single file. The modules will look for the expected state files inside that directory (e.g. `env_state_dict.pt`, `encoder_state_dict.pt`).

## Training - discrete environment model

```python
from deepcubeai.training.train_env_disc import TrainEnvDiscConfig, run_train_env_disc

cfg = TrainEnvDiscConfig(
    env="sokoban",
    train_data="deepcubeai/data/sokoban/offline/train_data.pkl",
    val_data="deepcubeai/data/sokoban/offline/val_data.pkl",
    nnet_name="sokoban_disc",
    save_dir="deepcubeai/saved_env_models",
    env_coeff=0.01,
    lr=1e-3,
    max_itrs=40000,
    batch_size=100,
)
run_train_env_disc(cfg)
```

Key module: `deepcubeai.training.train_env_disc`, dataclass `TrainEnvDiscConfig` and `run_train_env_disc(cfg)` are provided.

## Training - continuous environment model

```python
from deepcubeai.training.train_env_cont import TrainEnvContConfig, run_train_env_cont

cfg = TrainEnvContConfig(
    env="cube3",
    train_data="deepcubeai/data/cube3/offline/train_data.pkl",
    val_data="deepcubeai/data/cube3/offline/val_data.pkl",
    nnet_name="cube3_cont",
    save_dir="deepcubeai/saved_env_models",
    lr=1e-3,
    max_itrs=40000,
    batch_size=100,
    num_steps=1,
)
run_train_env_cont(cfg)
```

Key module: `deepcubeai.training.train_env_cont`, dataclass `TrainEnvContConfig` and `run_train_env_cont(cfg)`.

## Heuristic Q-learning

```python
from deepcubeai.training.qlearning import QLearningConfig, run_qlearning

cfg = QLearningConfig(
    env="cube3",
    env_model="deepcubeai/saved_env_models/cube3_disc",
    train="deepcubeai/data/cube3/offline_enc/train_data_enc.pkl",
    val="deepcubeai/data/cube3/offline_enc/val_data_enc.pkl",
    per_eq_tol=100,
    batch_size=1000,
    states_per_update=50_000_000,
    max_solve_steps=30,
    start_steps=30,
    goal_steps=30,
    num_test=1000,
    nnet_name="cube3_heur",
    save_dir="deepcubeai/saved_heur_models",
)
run_qlearning(cfg)
```

Key module: `deepcubeai.training.qlearning`, dataclass `QLearningConfig` and `run_qlearning(cfg)`.

## Heuristic Q-learning (distributed)

```python
from deepcubeai.training.qlearning_dist import QLearningDistConfig, run_qlearning_dist

cfg = QLearningDistConfig(
    env="cube3",
    env_model="deepcubeai/saved_env_models/cube3_disc",
    train="deepcubeai/data/cube3/offline_enc/train_data_enc.pkl",
    val="deepcubeai/data/cube3/offline_enc/val_data_enc.pkl",
    per_eq_tol=100,
    batch_size=1000,
    states_per_update=50_000_000,
    max_solve_steps=30,
    start_steps=30,
    goal_steps=30,
    num_test=1000,
    nnet_name="cube3_heur_dist",
    save_dir="deepcubeai/saved_heur_models",
)
run_qlearning_dist(cfg)  # Launch via torchrun for multi-process scaling
```

Launch (single node, 4 GPUs):

```bash
torchrun --nproc_per_node=4 -m deepcubeai train_heur --env cube3 --env_model_name cube3_disc \
    --heur_nnet_name cube3_heur_dist --heur_batch_size 1000 --states_per_update 50000000 \
    --max_solve_steps 30 --start_steps 30 --goal_steps 30 --per_eq_tol 100 --use_dist
```

Key module: `deepcubeai.training.qlearning_dist`. Every rank both generates data and trains. Uses NCCL when GPUs are present, Gloo otherwise.

## Search - Q* (and UCS via the same module)

```python
from deepcubeai.search_methods.qstar_imag import QStarImagConfig, run_qstar_imag

cfg = QStarImagConfig(
    env="cube3",
    states="deepcubeai/data/cube3/search_test/search_test_data.pkl",
    env_model="deepcubeai/saved_env_models/cube3_disc",
    results_dir="deepcubeai/results/cube3/experiment_qstar",
    per_eq_tol=100,
    weight=0.6,
    h_weight=1.0,
    batch_size=1,
    save_imgs=False,
)
run_qstar_imag(cfg)
```

Key module: `deepcubeai.search_methods.qstar_imag`, dataclass `QStarImagConfig` and `run_qstar_imag(cfg)`. When `h_weight == 0.0` the module performs Uniform Cost Search (UCS). Other flags (e.g., `nnet_batch_size`, `start_idx`, `verbose`, `debug`) are available.

Other search helpers: `deepcubeai.search_methods.ucs_imag` exposes `run_ucs_imag(cfg)` with a similar config dataclass. `gbfs_imag` provides GBFS utilities.

## Encoding offline datasets (env encoder → encoded offline)

```python
from deepcubeai.scripts.encode_offline_data import EncodeOfflineConfig, run_encode_offline

cfg = EncodeOfflineConfig(
    env="cube3",
    env_dir="deepcubeai/saved_env_models/cube3_disc",
    data="deepcubeai/data/cube3/offline/train_data.pkl",
    data_enc="deepcubeai/data/cube3/offline_enc/train_data_enc.pkl",
)
run_encode_offline(cfg)
```

Key module: `deepcubeai.scripts.encode_offline_data`, dataclass `EncodeOfflineConfig` and `run_encode_offline(cfg)`.

## Generating offline datasets

```python
from deepcubeai.scripts.generate_offline_data import GenerateOfflineConfig, run_generate_offline

cfg = GenerateOfflineConfig(
    env="cube3",
    num_episodes=2000,
    num_steps=20,
    data_file="deepcubeai/data/cube3/offline/train_data.pkl",
    num_procs=4,
)
run_generate_offline(cfg)
```

Key module: `deepcubeai.scripts.generate_offline_data`, dataclass `GenerateOfflineConfig` and `run_generate_offline(cfg)`.

## Offline dataset visualization

```python
from deepcubeai.extra.offline_data_viz import OfflineDataVizConfig, run_offline_data_viz

cfg = OfflineDataVizConfig(
    env="sokoban",
    train_data="deepcubeai/data/sokoban/offline/train_data.pkl",
    val_data="deepcubeai/data/sokoban/offline/val_data.pkl",
    num_train_trajs=8,
    num_train_steps=2,
    save_imgs="sample_images",
)
run_offline_data_viz(cfg)
```

Key module: `deepcubeai.extra.offline_data_viz`, dataclass `OfflineDataVizConfig` and `run_offline_data_viz(cfg)`.

## Plotting

- `deepcubeai.extra.plot_disc_vs_cont` → `PlotDiscVsContConfig`, `run_plot_disc_vs_cont(cfg)` (compare disc vs cont MSE)

## Pipeline-level usage and introspection

The repository also exposes a central `PipelineConfig` that the CLI uses to build end-to-end pipelines. You can construct and serialize it:

```python
from deepcubeai.config import PipelineConfig
cfg = PipelineConfig(env='cube3', env_model_name='cube3_disc')
print(cfg.to_json_bytes())  # compact JSON bytes
```

Pipeline orchestration helpers live in `deepcubeai.pipeline` and call the module-level `run_*` functions above.
