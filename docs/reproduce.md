# Reproducing the paper results

This document explains the contents of the `reproduce_results` helper folder and how to reproduce the paper results either locally (direct run) or on a cluster using the included SLURM scripts.

Paths referenced here are relative to the repository root.

## What is in `reproduce_results/`

- `run_directly/`
  - `get_cpu_num.sh` - small helper to detect CPU count (used by some reproduce scripts).
  - `reproduce_cube3.sh` - script to reproduce Cube3 results.
  - `reproduce_digitjump.sh` - script to reproduce DigitJump results.
  - `reproduce_iceslider.sh` - script to reproduce IceSlider results.
  - `reproduce_sokoban.sh` - script to reproduce Sokoban results.

- `SLURM_scripts/`
  - `submit.sh` - general SLURM submission wrapper for experiments.
  - `submit_ddp_heur.sh` - example SLURM submission for distributed heuristics training.

## High-level contract

- Inputs: repository root, an appropriate Python environment (see `README.md` / `docs/installation.md`), and any saved models or datasets expected by the scripts (the repo provides `saved_env_models/` and `saved_heur_models/`).
- Outputs: experiment logs and numeric/plot results stored under `results/` (and cluster logs under `reproduce_results/SLURM_scripts/job_run_outputs/` in example runs).

Assumption: You have installed the project's dependencies (via Pixi, conda, or pip) and activated the environment before running the scripts. If not, see `README.md` -> "Install" or `docs/installation.md`.

## Reproduce locally

Use the scripts in `reproduce_results/run_directly/` to re-run experiments on your machine. These scripts are small wrappers that call the repo CLI or Python entrypoints with fixed arguments to reproduce the paper experiments.

1. Activate your environment (example using Pixi):

```bash
# from repo root
pixi shell
```

1. Make the run scripts executable:

```bash
chmod +x reproduce_results/run_directly/*.sh
```

1. Run a reproduction script. Example for Cube3:

```bash
bash reproduce_results/run_directly/reproduce_cube3.sh
```

Similarly, for other environments:

```bash
bash reproduce_results/run_directly/reproduce_iceslider.sh
bash reproduce_results/run_directly/reproduce_sokoban.sh
bash reproduce_results/run_directly/reproduce_digitjump.sh
```

> [!NOTE]
>
> - If your system has GPUs available, ensure GPUs are visible to the environment (e.g., drivers installed, CUDA_VISIBLE_DEVICES set properly).
> - If a script fails due to missing saved models, point it to the correct folder under `saved_env_models/` or `saved_heur_models/` (edit the `.sh` script to adjust paths or pass the appropriate CLI flags).

## Reproduce on a SLURM cluster

The `SLURM_scripts/` folder contains example submission scripts for running experiments on an HPC.

1. Inspect and edit the submission script to match your cluster's partition, account, nodes, and resource limits.
2. Submit the script with `sbatch` (after editing any cluster-specific fields):

    ```bash
    sbatch reproduce_results/SLURM_scripts/submit_iceslider_ddp_heur.sh
    ```

3. Your cluster scheduler will create job-specific `.out` and `.err` files in the paths configured by the SLURM script.

## Where results and models are expected/saved

- Saved environment models: `saved_env_models/`
- Saved heuristic models: `saved_heur_models/`
- Experiment numerical/plot results: `results/`

If a reproduce script expects models not present in the repo, you will need to run the training stage first (see `training/` and `docs/usage.md` for details).
