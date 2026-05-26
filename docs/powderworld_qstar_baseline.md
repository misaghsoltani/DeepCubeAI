# Powderworld Q* Macro Baseline

This runbook launches the DeepCube-style Q* macro baseline for OGBench Powderworld
`easy`, `medium`, and `hard`.

The macro abstraction plans on an 8x8 grid. One macro action chooses an element
and an `(x, y)` brush cell:

- `easy`: 2 drawable elements, 128 actions.
- `medium`: 5 drawable elements, 320 actions.
- `hard`: 8 drawable elements, 512 actions.

For `medium` and `hard`, this is a planning relaxation: the learned/symbolic
macro model sets cells directly and does not model Powderworld physics such as
falling sand, water, fire, gas, or ice interactions. Always report both latent
Q* results and real OGBench rollout results.

## Clone-Only Setup

Do not copy the working folders. The large files live under `deepcubeai/data`,
`deepcubeai/saved_*`, and `deepcubeai/results`; these are ignored by git and can
be regenerated.

Clone the code repositories next to each other:

```bash
mkdir -p ~/projects/powderworld-qstar
cd ~/projects/powderworld-qstar

git clone <your-deepcubeai-fork-or-branch-url> DeepCubeAI
git clone https://github.com/WJ2003B/mqe-release.git mqe-release
```

`mqe-release` is small and is only used here as the local OGBench Powderworld
provider. We add it to `PYTHONPATH`; this avoids installing the full OGBench
training stack.

## Dependencies With uv

Use Python 3.11-3.12 for the cleanest install. The local laptop may also work
with newer Python, but large runs should use a conventional CUDA PyTorch setup.

From `DeepCubeAI`:

```bash
cd DeepCubeAI
uv python install 3.12
uv sync --python 3.12

export PYTHONPATH="$PWD:$(cd ../mqe-release && pwd):${PYTHONPATH:-}"
uv run python - <<'PY'
import torch
import gymnasium
import ogbench.powderworld  # noqa: F401
print(torch.__version__, torch.cuda.is_available())
PY
```

Running this baseline does not require JAX, Flax, Distrax, or WandB.

If `uv sync` installs an unsuitable PyTorch build for the target GPU, install the
matching PyTorch wheel in the `.venv` following the official PyTorch command for
that machine.

## One-Command Launcher

The generic launcher accepts `DIFF=easy|medium|hard` and
`MODE=oracle|neural`.

Oracle smoke run:

```bash
USE_UV=1 DIFF=medium MODE=oracle QUICK=1 \
  bash reproduce_results/run_directly/run_powderworld_macro_qstar.sh
```

Learned baseline:

```bash
USE_UV=1 DIFF=hard MODE=neural QUICK=0 FRESH=1 \
  bash reproduce_results/run_directly/run_powderworld_macro_qstar.sh
```

For staged runs:

```bash
USE_UV=1 DIFF=hard MODE=neural STAGES=train_encoder,train_models \
  bash reproduce_results/run_directly/run_powderworld_macro_qstar.sh

USE_UV=1 DIFF=hard MODE=neural STAGES=gen_search,qstar,real_eval \
  bash reproduce_results/run_directly/run_powderworld_macro_qstar.sh
```

For `MODE=oracle`, use `STAGES=oracle_ckpt,gen_search,qstar,real_eval` if you
do not want the default `all`.

## Oracle Smoke Baseline

This creates deterministic symbolic encoder/model/heuristic checkpoints and is
the quickest way to test search plumbing.

```bash
DIFF=medium

python deepcubeai/scripts/create_powderworld_macro_oracle.py \
  --env powderworld_${DIFF}_macro \
  --env_model_dir deepcubeai/saved_env_models/powderworld_${DIFF}_macro_oracle \
  --heur_dir deepcubeai/saved_heur_models/powderworld_${DIFF}_macro_oracle/current

python deepcubeai/scripts/generate_search_test_data.py \
  --env powderworld_${DIFF}_macro \
  --num_episodes 200 \
  --start_level 0 \
  --data_file deepcubeai/data/powderworld_${DIFF}_macro/search_test/search_test_200.pkl

python deepcubeai/search_methods/qstar_imag.py \
  --env powderworld_${DIFF}_macro \
  --states deepcubeai/data/powderworld_${DIFF}_macro/search_test/search_test_200.pkl \
  --env_model deepcubeai/saved_env_models/powderworld_${DIFF}_macro_oracle \
  --heur deepcubeai/saved_heur_models/powderworld_${DIFF}_macro_oracle/current \
  --results_dir deepcubeai/results/powderworld_${DIFF}_macro/qstar_oracle_200 \
  --per_eq_tol 98.9 \
  --batch_size 1 \
  --nnet_batch_size 4096 \
  --weight 1.0

python deepcubeai/scripts/eval_powderworld_macro_qstar_real.py \
  --difficulty ${DIFF} \
  --results deepcubeai/results/powderworld_${DIFF}_macro/qstar_oracle_200/results.pkl \
  --out deepcubeai/results/powderworld_${DIFF}_macro/qstar_oracle_200/real_env_eval.json
```

Suggested `--per_eq_tol`:

- `easy`: `100`
- `medium`: `98.9` for strict 2-cell tolerance, or `97.9` for 4-cell tolerance
- `hard`: `99.3` for roughly 2 cells, `98.6` for 4 cells, `97.9` for 6 cells

## Learned Baseline

Train the image-to-grid encoder first. Then train the macro transition and
heuristic models. The checkpoints are difficulty-specific; do not reuse an
`easy` encoder for `medium` or `hard`.

```bash
DIFF=hard

python deepcubeai/scripts/train_powderworld_macro_encoder.py \
  --difficulty ${DIFF} \
  --num_train_pairs 2000 \
  --num_val_pairs 300 \
  --iters 2000 \
  --batch_size 256 \
  --eval_batch_size 1024

python deepcubeai/scripts/train_powderworld_macro_models.py \
  --difficulty ${DIFF} \
  --iters 2000 \
  --batch_size 2048 \
  --eval_batch_size 4096 \
  --eval_batches 4

python deepcubeai/scripts/generate_search_test_data.py \
  --env powderworld_${DIFF}_macro \
  --num_episodes 200 \
  --start_level 0 \
  --data_file deepcubeai/data/powderworld_${DIFF}_macro/search_test/search_test_200.pkl

python deepcubeai/search_methods/qstar_imag.py \
  --env powderworld_${DIFF}_macro_neural \
  --states deepcubeai/data/powderworld_${DIFF}_macro/search_test/search_test_200.pkl \
  --env_model deepcubeai/saved_env_models/powderworld_${DIFF}_macro_neural \
  --heur deepcubeai/saved_heur_models/powderworld_${DIFF}_macro_neural/current \
  --results_dir deepcubeai/results/powderworld_${DIFF}_macro/qstar_neural_200 \
  --per_eq_tol 97.9 \
  --batch_size 1 \
  --nnet_batch_size 4096 \
  --weight 1.0

python deepcubeai/scripts/eval_powderworld_macro_qstar_real.py \
  --difficulty ${DIFF} \
  --results deepcubeai/results/powderworld_${DIFF}_macro/qstar_neural_200/results.pkl \
  --out deepcubeai/results/powderworld_${DIFF}_macro/qstar_neural_200/real_env_eval.json
```

Use `CUDA_VISIBLE_DEVICES=...` to choose a GPU. The training scripts default to
CUDA when PyTorch sees a GPU.
