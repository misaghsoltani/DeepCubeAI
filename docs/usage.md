# Usage

## Configuration

Configuration is merged from three sources (lowest → highest precedence):

1. JSON file specified with `--config-file`
2. Environment variables prefixed with `DCAI_` (see note below)
3. CLI arguments passed to the stage command

Examples:

```bash
# set values via environment
export DCAI_ENV=cube3
export DCAI_ENV_MODEL_NAME=cube3_disc
export DCAI_NUM_OFFLINE_STEPS=100

# run pipeline stage (CLI overrides env)
deepcubeai gen_offline --num_val_eps 1000 --env cube3
```

> [!NOTE]
>
> - The prefix is `DCAI_` followed by the uppercase field name from `PipelineConfig` (for example `env_model_name` -> `DCAI_ENV_MODEL_NAME`).
> - Values are parsed into the annotated type where possible (int, float, bool, Path-like, string). Unknown/extra keys in a JSON config are ignored.
> - Boolean truthy values accepted (case-insensitive): `1`, `true`, `yes`, `y`, `on`.

## Pipeline stages

Typical pipeline flow (one common ordering):

1. `gen_offline` - generate training & validation trajectories
2. `train_model_disc` or `train_model_cont` - train environment model
3. `gen_env_test` - generate held-out trajectories for model evaluation
4. `test_model` or `test_model_cont` - evaluate the trained env model
5. `encode_offline` - produce encoded offline datasets for heuristic training
6. `train_heur` - train the heuristic network (DQN+HER)
7. `gen_search_test` - create start/goal pairs for search
8. `qstar` / `ucs` / `gbfs` - run search evaluations

Get stage-specific help with:

```bash
deepcubeai <stage> -h
```

## Environment registry

The pipeline includes a small registry of environments. Commands:

- List known environments (human readable):

```bash
deepcubeai envs
```

- JSON output:

```bash
deepcubeai envs --json
```

- Add a user environment to the registry:

```bash
deepcubeai envs-add --key myenv --module mypkg.my_env_module --attr MyEnvClass
```

- Remove a previously added environment:

```bash
deepcubeai envs-remove --key myenv
```

## Data and output layout

Default data root: `deepcubeai/data/<env>` when `--data_dir` is omitted. You can change the base with `--data_dir` or set `DCAI_DATA_DIR`.

Path resolution rules (exact):

- If `--data_dir` (or `DCAI_DATA_DIR`) is an absolute path, it is used as-is.
- If it starts with `deepcubeai/data/` it is interpreted relative to the repository root.
- Otherwise it resolves to `deepcubeai/data/<data_dir>` under the package directory.

Typical layout produced by stages:

```text
deepcubeai/data/<env>/offline/           # raw train/val pickles
deepcubeai/data/<env>/offline_enc/       # encoded datasets for heuristics
deepcubeai/data/<env>/model_test/        # held-out trajectories for model testing
deepcubeai/data/<env>/search_test/       # start/goal pairs for search evaluation
deepcubeai/data/<env>/sample_images/     # images rendered from datasets (visualize_data)
deepcubeai/saved_env_models/<model_name>/
deepcubeai/saved_heur_models/<heur_name>/current/
deepcubeai/results/<env>/                # search/experiment outputs
```

Key files and naming rules:

- Offline (defaults): `train_data.pkl`, `val_data.pkl` under the `offline/` directory.
- Encoded: `train_data_enc.pkl`, `val_data_enc.pkl` under `offline_enc/`.
- Env test: `env_test_data.pkl` under `model_test/`.
- Search test: `search_test_data.pkl` under `search_test/`.

If you pass `--data_file_name <NAME>` then the code will use `NAME` as the base. If the provided string already contains the expected suffix (for example `train_data`), it is used as-is; otherwise the pipeline appends the appropriate suffix (for example `"<NAME>_train_data.pkl"`).

## Running Different Stages

This section flags, derived paths, and outputs for each pipeline stage. Relative paths resolve under `deepcubeai/data/<data_dir>` unless you supply an absolute path. If `--data_dir` is omitted it defaults to the environment key (`--env`).

Common options

- `--data_dir`: base directory name/path for data. If not absolute and not starting with `deepcubeai/data/`, it will be interpreted as `deepcubeai/data/<data_dir>`.
- `--data_file_name`: base name used to compose file names (suffixes are automatically appended).

Automatic names (given `--data_file_name <N>`)

- Offline: `<N>_train_data.pkl`, `<N>_val_data.pkl`
- Env test: `<N>_env_test_data.pkl`
- Search test: `<N>_search_test_data.pkl`
- Encoded: `<N>_train_data_enc.pkl`, `<N>_val_data_enc.pkl`

## gen_offline

Generate offline train/val datasets.

Required:

- `--env`
- `--num_offline_steps` (steps per episode)

Optional:

- `--data_dir`, `--data_file_name`, `--num_cpus`
- Episode counts: `--num_train_eps`, `--num_val_eps` (if both omitted the pipeline defaults to 9000/1000)
- Level seeding: `--start_level`, `--num_levels` (see `derive_seeds` semantics)

Outputs:

- `deepcubeai/data/<data_dir>/offline/<name>.pkl` (train/val files; base names are `train_data`/`val_data` unless you override `--data_file_name`)

## gen_env_test

Generate trajectories for model evaluation.

Required:

- `--env`
- `--num_offline_steps`

Optional:

- `--num_test_eps` (default 100), `--num_cpus`, `--data_dir`, `--data_file_name`, seeding flags.

Output:

- `deepcubeai/data/<data_dir>/model_test/<name>.pkl` (default base `env_test_data` unless overridden)

## gen_search_test

Generate start/goal pairs for search.

Required:

- `--env`

Optional:

- `--num_test_eps` (default 100), `--num_offline_steps` (optional; default -1), seeding flags, `--data_dir`, `--data_file_name`.
- `--reverse` - cube3-only helper that builds pairs by starting from the canonical goal and scrambling (only meaningful for `cube3`).

Output:

- `deepcubeai/data/<data_dir>/search_test/<name>.pkl` (default base `search_test_data` unless overridden)

## train_model_disc (discrete)

Train the discrete environment model. The pipeline runs a sequence of scheduled training phases (see code for exact schedules).

Required:

- `--env`

Important options:

- `--env_model_name` - saved model directory under `deepcubeai/saved_env_models/` (defaults to `env_model`)
- `--env_batch_size` (default 100)
- `--print_interval` controls status printing frequency

Data is read from the `gen_offline` outputs unless you point to custom files.

Typical outputs under `deepcubeai/saved_env_models/<env_model_name>/` include model checkpoint files, arguments (`args.pkl`), training state (`train_itr.pkl`), printed `output.txt` and a `pics/` folder with samples (exact filenames depend on the trainer implementation).

## train_model_cont (continuous)

Train a continuous environment model using schedules defined in the pipeline.

Outputs:

- `deepcubeai/saved_env_models/<env_model_name>/model_state_dict.pt` (and other artifacts produced by the trainer)

## test_model / test_model_cont

Evaluate env models on the held-out env test data (produced by `gen_env_test`).

Required:

- `--env`, `--env_model_name`

Options:

- `--print_interval` (default 1)
- Override test data location with `--data_dir` / `--data_file_name` or pass explicit paths in a JSON config.

Outputs:

- Reconstruction images and any evaluation artifacts typically written under the model directory (trainer scripts put images under `pics/` by convention).

## encode_offline

Encode train/val datasets using a trained environment model. The encoder scripts write encoded pickles used by heuristic training.

Required:

- `--env`, `--env_model_name`

Outputs:

- Encoded datasets under `offline_enc/` (default names `train_data_enc.pkl` / `val_data_enc.pkl` unless `--data_file_name` is provided).

## train_heur

Train the heuristic network (DQN+HER) using encoded datasets.

Required:

- `--env`, `--env_model_name`
- `--heur_nnet_name`
- `--heur_batch_size`, `--states_per_update`, `--max_solve_steps`, `--start_steps`, `--goal_steps`
- `--per_eq_tol` - percent of latent elements that must match to declare two latent states equal

Optional:

- `--num_test` (default 1000)
- `--use_dist` enables multi-process FSDP when launched under `torchrun` (recommended for multi-GPU / multi-node).

Outputs under `deepcubeai/saved_heur_models/<heur_nnet_name>/` typically include `current/` and `target/` checkpoint folders, `args.pkl`, and training logs such as `output.txt`.

## qstar

Run Q* search on the prepared search test states.

Key options:

- `--env_model_name`, `--heur_nnet_name` (heuristic name can be omitted to use the default `current` checkpoint)
- `--qstar_batch_size`, `--qstar_weight` (path cost weight w), `--qstar_h_weight` (heuristic weight h)
- `--per_eq_tol` (latent equality tolerance)
- `--save_imgs` / `--no_save_imgs` to toggle saving solution images
- `--qstar_results_dir` to override the default results subdirectory
- `--search_test_data` to point to a custom search test file

By default the pipeline writes results under `deepcubeai/results/<env>/<derived-name>/` (the derived name encodes model/heur/weights). Outputs include `output.txt`, `results.pkl`, and optionally solution images.

## ucs

Uniform-Cost Search implemented as Q* with `w=1, h=0`.

Options:

- `--ucs_batch_size`, `--per_eq_tol`, `--ucs_results_dir`, `--save_imgs`/`--no_save_imgs`, `--search_test_data`

## gbfs

Greedy Best-First Search using the learned heuristic.

Options:

- `--heur_nnet_name`, `--per_eq_tol`, `--search_itrs` (search iterations per state), `--gbfs_results_dir`, `--search_test_data`

## disc_vs_cont

Compare MSE of discrete vs continuous environment models and save comparison plots.

Required:

- `--env`
- `--env_model_dir_cont` (path to continuous model directory)

Optional:

- `--env_model_dir_disc` (or `--env_model_name` to derive the discrete model path), `--num_episodes`, `--num_steps`, `--save_dir`

## visualize_data

Render sample images from offline datasets for quick visual inspection. Images are written to `deepcubeai/data/<data_dir>/sample_images/` by default.
