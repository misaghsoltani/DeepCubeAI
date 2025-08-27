# CLI Reference

DeepCubeAI provides a stage-based command line interface implemented in `deepcubeai/pipeline.py`.

## Quick Start

```bash
# Using the console script
deepcubeai gen_offline --env cube3 --num_offline_steps 30 --num_train_eps 9000 --num_val_eps 1000

# Or equivalently via the module entrypoint
python -m deepcubeai gen_offline --env cube3 --num_offline_steps 30
```

Get contextual help (lists only that stage's flags):

```bash
deepcubeai <stage> --help # or -h
```

## Stages

| Stage             | Purpose                                               |
| ----------------- | ----------------------------------------------------- |
| gen_offline       | Generate offline train/val datasets                   |
| gen_env_test      | Generate offline test data for env model evaluation   |
| gen_search_test   | Generate start/goal states for search evaluation      |
| train_model_disc  | Train the discrete environment model                  |
| train_model_cont  | Train the continuous environment model                |
| test_model        | Evaluate a discrete env model                         |
| test_model_cont   | Evaluate a continuous env model                       |
| encode_offline    | Encode offline train/val with the trained env model   |
| train_heur        | Train the heuristic network                           |
| qstar             | Run Q* search                                         |
| ucs               | Run Uniform-Cost Search (Q* with w=1, h=0)            |
| gbfs              | Run Greedy Best-First Search                          |
| disc_vs_cont      | Compare discrete vs continuous env models (MSE plots) |
| visualize_data    | Render and save sample offline data images            |
| compare_solutions | Compare two solution result pickles                   |
| envs              | List known environments                               |
| envs-add          | Add a user environment to the registry                |
| envs-remove       | Remove a user environment from the registry           |

## Configuration & precedence

Configuration is merged into a single `PipelineConfig` with this precedence (later sources override earlier ones):

1. JSON file passed via `--config-file`
2. Environment variables (prefix `DCAI_`, e.g. `DCAI_ENV=cube3`)
3. CLI arguments (stage-specific flags)

Notes about merging and parsing:

- Booleans use logical OR when merging. If any source enables a boolean flag, it stays enabled.
- Other fields: the last non-empty value wins (CLI > env > file).
- Environment variable values are parsed to the annotated type where possible (int/float/bool/paths). Unknown values fall back to strings.

Example config file (`config.json`):

```json
{
 "env": "cube3",
 "data_dir": "cube3",
 "env_model_name": "cube3_disc",
 "heur_nnet_name": "cube3_heur",
 "per_eq_tol": 100
}
```

Usage:

```bash
deepcubeai train_model_disc --config-file config.json --env_batch_size 200
```

Environment variable example (bash):

```bash
export DCAI_ENV=cube3
export DCAI_ENV_MODEL_NAME=cube3_disc
deepcubeai test_model --env_model_name cube3_disc  # CLI has precedence
```

## File & directory conventions

Unless you provide absolute paths, data and model directories are rooted under the package tree (the `deepcubeai` package root). The default locations are:

- Offline data: `deepcubeai/data/<data_dir>/offline/`
- Encoded data: `deepcubeai/data/<data_dir>/offline_enc/`
- Env test: `deepcubeai/data/<data_dir>/model_test/`
- Search test: `deepcubeai/data/<data_dir>/search_test/`
- Sample images: `deepcubeai/data/<data_dir>/sample_images/`
- Env models: `deepcubeai/saved_env_models/<env_model_name>/`
- Heuristic models (current checkpoint): `deepcubeai/saved_heur_models/<heur_nnet_name>/current/`

These defaults are produced by `PipelineConfig.offline_paths()`, `env_model_paths()` and `heur_model_paths()`.

### Automatic filenames

If you pass `--data_file_name <N>`, the pipeline composes filenames by appending logical suffixes unless the provided name already contains the suffix. The resulting files have `.pkl` extensions. The mappings are:

| Logical file  | Produced filename (on disk) |
| ------------- | --------------------------- |
| train offline | `<N>_train_data.pkl`        |
| val offline   | `<N>_val_data.pkl`          |
| env test      | `<N>_env_test_data.pkl`     |
| search test   | `<N>_search_test_data.pkl`  |
| encoded train | `<N>_train_data_enc.pkl`    |
| encoded val   | `<N>_val_data_enc.pkl`      |

If you embed the suffix in `--data_file_name` (for example `--data_file_name my_train_data`), the suffix is not appended again.

## Typical end-to-end flow

1. `gen_offline`
2. `train_model_disc` or `train_model_cont`
3. `test_model` / `test_model_cont`
4. `encode_offline`
5. `train_heur`
6. `gen_search_test`
7. `qstar` / `ucs` / `gbfs`
8. `disc_vs_cont`, `visualize_data`

## Examples

Train discrete env model :

```bash
deepcubeai train_model_disc --env cube3 --data_file_name 10k_stp30 --env_model_name cube3_disc
```

Encode and train heuristic:

```bash
deepcubeai encode_offline --env cube3 --data_file_name 10k_stp30 --env_model_name cube3_disc
deepcubeai train_heur --env cube3 --data_file_name 10k_stp30 --env_model_name cube3_disc \
 --heur_nnet_name cube3_heur --per_eq_tol 100 --heur_batch_size 10000 \
 --states_per_update 50000000 --max_solve_steps 30 --start_steps 30 --goal_steps 30
```

Run Q* search:

```bash
deepcubeai qstar --env cube3 --env_model_name cube3_disc --heur_nnet_name cube3_heur \
 --qstar_weight 0.6 --qstar_h_weight 1.0 --qstar_batch_size 10000 --per_eq_tol 100 --save_imgs
```

### Notes and useful flags

- `--config-file` accepts a JSON object. Unknown keys are ignored. Use it to store long-running pipelines or to share config between runs.
- Environment variables use the `DCAI_` prefix and map to dataclass fields (uppercased). Values are parsed to ints/floats/bools/paths when possible.
- `gen_search_test` has a `--reverse` flag that is only meaningful for `cube3` (it reverses start/goal generation).
- `train_heur` requires several required flags: `--heur_nnet_name`, `--heur_batch_size`, `--states_per_update`, `--max_solve_steps`, `--start_steps`, `--goal_steps`, and `--per_eq_tol`. `--num_test` defaults to 1000 if not provided.
- `disc_vs_cont` requires `--env_model_dir_cont` (path to a continuous model directory) unless you explicitly provide it.

- `gen_offline` and `gen_env_test` require `--num_offline_steps` (steps per episode to generate).
- `gen_search_test` has a `--reverse` flag that is only meaningful for `cube3` (it reverses start/goal generation).
- `train_heur` requires several required flags: `--heur_nnet_name`, `--heur_batch_size`, `--states_per_update`, `--max_solve_steps`, `--start_steps`, `--goal_steps`, and `--per_eq_tol`. `--num_test` defaults to 1000 if not provided.
- `disc_vs_cont` requires `--env_model_dir_cont` (path to a continuous model directory).
