
# SLURM & Distributed Heuristic Training

This document describes how the repository's distributed heuristic training is wired and how to run it on single-node and multi-node clusters. The contents below match the actual implementation and the example SLURM submit script `reproduce_results/SLURM_scripts/submit_ddp_heur.sh`.

## Quick summary

- PyTorch multi-process distributed training (preferred: `torchrun`).
- `dist_utils.get_env_vars()` and `dist_utils.setup_ddp()` are used by the training entry point (`qlearning_dist.py`) to detect environment variables and initialize the process group.
- The generation pipeline (`update_utils_dist.generate_batches_gpu`) produces GPU-resident batches using a small ring buffer (argument: `--ring`) and CUDA streams/events to overlap generation with training.

## Environment variables

The code works with standard torchrun / SLURM environment variables. The helper `dist_utils.get_env_vars()` reads these in the following priority:

- torchrun / torch.distributed.run: `RANK`, `WORLD_SIZE`, `LOCAL_RANK` (preferred)
- SLURM: `SLURM_PROCID`, `SLURM_NTASKS`, and optionally `SLURM_GPUS_ON_NODE` or `SLURM_GPUS`

The SLURM helper script `submit_ddp_heur.sh` sets/uses:

- `MASTER_ADDR` and `MASTER_PORT` (rendezvous)
- `WORLD_SIZE`, `NNODES`, `GPUS_PER_NODE`, and per-node `LOCAL_RANK` via torchrun

dist_utils.setup_ddp(...) will set up the backend (`nccl` if CUDA is available else `gloo`) and call `init_process_group(...)`. When using NCCL it sets the CUDA device early via `torch.cuda.set_device(local_rank % torch.cuda.device_count())` so the process group can initialize cleanly.

## How to launch (`submit_ddp_heur.sh`)

The provided SLURM submit wrapper composes `TORCHRUN_ARGS` and `TRAIN_ARGS` and finally runs:

```bash
srun torchrun "${TORCHRUN_ARGS[@]}" "${TRAIN_ARGS[@]}"
```

Example `torchrun` options the script uses:

- `--nnodes=$NNODES`
- `--nproc_per_node=$GPUS_PER_NODE`
- `--rdzv-backend=c10d`
- `--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT`
- `--rdzv-id=$SLURM_JOB_ID`

Example training invocation the submit script builds (flag names used in the script):

- `-m deepcubeai train_heur` (module entry point)
- `--env <ENV_NAME>` (environment key)
- `--data_dir <ENV_NAME>` and `--data_file_name <NAME>` (where offline states live)
- `--env_model_name <env_model_dir>` (directory containing `env_state_dict.pt`)
- `--heur_nnet_name <model_name>` (model save/log prefix)
- `--per_eq_tol <int>` (percent equal threshold for solved check)
- `--heur_batch_size <int>` (logical global batch size)
- `--states_per_update <int>` (how many generated states to process before checking updates)
- `--start_steps`, `--goal_steps`, `--max_solve_steps` (generation/solve depth)
- `--num_test` (validation sample size)
- `--use_dist` (enable distributed behavior in the train command)
- `--lr`, `--lr_d`, `--max_itrs`, `--update_nnet_batch_size` (learning and update params)
- `--ring` (ring-buffer slots for asynchronous generation)
- `--amp`, `--compile`, `--compile_all_models` (optional runtime flags)

Note: the script conditionally appends `--amp`, `--compile`, and `--compile_all_models` based on user-configurable shell variables.

## Important implementation details

- Rank detection: `qlearning_dist.py` calls `dist_utils.get_env_vars()` to compute rank/world/local ranks.

- Process group init: `dist_utils.setup_ddp(...)` prepares `MASTER_ADDR`, `MASTER_PORT` and calls `init_process_group(...)` choosing `nccl` when CUDA is available. It also tries to provide `device_id` for newer PyTorch versions and falls back safely for older ones.
- Device selection: when CUDA/NCCL is used the code sets the current CUDA device from `local_rank` before initializing the process group.
- Asynchronous generation: `update_utils_dist.generate_batches_gpu(...)` runs generation on a separate CUDA stream and uses a small ring of `RingSlot` objects with `torch.cuda.Event` objects (`consumed_ev`, `ready_ev`) to overlap generation and training. The number of slots is controlled via the `--ring` flag.
- Ring-buffer behavior: the ring slots hold tensors that live on GPU. Generator stream fills slots and signals readiness via events. The training stream waits on those events and consumes data, allowing generation and training to overlap.
- Sampling and Boltzmann action selection: `update_utils_dist.sample_boltzmann(...)` samples actions from a Boltzmann distribution over Q-values (used during generation).
- DCP checkpointing: `qlearning_dist.py` builds a DCP-friendly state dict using `get_model_state_dict(...)` and saves shards via `state_dict_saver.save(...)`. Optionally the script consolidates shards into a single `torch.save` file on rank 0.

## Where to look in the code

- `deepcubeai/training/qlearning_dist.py` - training entry point, argument parsing, distributed init, model loading, DCP checkpoint helpers.
- `deepcubeai/utils/dist_utils.py` - `get_env_vars()` and `setup_ddp()` used by the training script.
- `deepcubeai/utils/update_utils_dist.py` - GPU-resident sample generation, `generate_batches_gpu()` and `random_walk_gpu()` implement ring-buffer + stream overlap.
- `reproduce_results/SLURM_scripts/submit_ddp_heur.sh` - example SLURM wrapper that composes `torchrun` and training args.
