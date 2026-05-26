#!/usr/bin/env bash
# Generic Powderworld macro Q* launcher.
#
# Examples from the DeepCubeAI repo root:
#   USE_UV=1 DIFF=medium MODE=oracle QUICK=1 bash reproduce_results/run_directly/run_powderworld_macro_qstar.sh
#   USE_UV=1 DIFF=hard MODE=neural QUICK=0 FRESH=1 bash reproduce_results/run_directly/run_powderworld_macro_qstar.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DCAI_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="$(cd "${DCAI_DIR}/.." && pwd)"
MQE_DIR="${MQE_DIR:-${PROJECT_DIR}/mqe-release}"

if [[ ! -d "${MQE_DIR}" ]]; then
  echo "Could not find mqe-release at MQE_DIR=${MQE_DIR}" >&2
  echo "Clone it next to DeepCubeAI or set MQE_DIR=/path/to/mqe-release." >&2
  exit 1
fi

export PYTHONPATH="${DCAI_DIR}:${MQE_DIR}:${PYTHONPATH:-}"
cd "${DCAI_DIR}"

DIFF="${DIFF:-medium}"
MODE="${MODE:-neural}"
USE_UV="${USE_UV:-0}"
QUICK="${QUICK:-1}"
STAGES="${STAGES:-all}"
DRY_RUN="${DRY_RUN:-0}"
FRESH="${FRESH:-0}"

case "${DIFF}" in
  easy)
    DEFAULT_PER_EQ_TOL="100"
    ;;
  medium)
    DEFAULT_PER_EQ_TOL="98.9"
    ;;
  hard)
    DEFAULT_PER_EQ_TOL="97.9"
    ;;
  *)
    echo "DIFF must be one of: easy, medium, hard" >&2
    exit 1
    ;;
esac

case "${MODE}" in
  oracle|neural)
    ;;
  *)
    echo "MODE must be one of: oracle, neural" >&2
    exit 1
    ;;
esac

if [[ "${QUICK}" == "1" ]]; then
  NUM_TRAIN_PAIRS="${NUM_TRAIN_PAIRS:-1000}"
  NUM_VAL_PAIRS="${NUM_VAL_PAIRS:-200}"
  ENCODER_ITRS="${ENCODER_ITRS:-800}"
  MODEL_ITRS="${MODEL_ITRS:-1000}"
  MODEL_BATCH="${MODEL_BATCH:-4096}"
  SEARCH_TEST_EPS="${SEARCH_TEST_EPS:-50}"
  SEARCH_START_LEVEL="${SEARCH_START_LEVEL:-0}"
else
  NUM_TRAIN_PAIRS="${NUM_TRAIN_PAIRS:-5000}"
  NUM_VAL_PAIRS="${NUM_VAL_PAIRS:-1000}"
  ENCODER_ITRS="${ENCODER_ITRS:-2000}"
  MODEL_ITRS="${MODEL_ITRS:-2000}"
  MODEL_BATCH="${MODEL_BATCH:-8192}"
  SEARCH_TEST_EPS="${SEARCH_TEST_EPS:-200}"
  SEARCH_START_LEVEL="${SEARCH_START_LEVEL:-1000}"
fi

TRAIN_SEED="${TRAIN_SEED:-0}"
VAL_SEED="${VAL_SEED:-100000}"
ENCODER_BATCH="${ENCODER_BATCH:-256}"
MODEL_EVAL_BATCH="${MODEL_EVAL_BATCH:-4096}"
MODEL_EVAL_BATCHES="${MODEL_EVAL_BATCHES:-4}"
PER_EQ_TOL="${PER_EQ_TOL:-${DEFAULT_PER_EQ_TOL}}"
QSTAR_BATCH="${QSTAR_BATCH:-1}"
QSTAR_WEIGHT="${QSTAR_WEIGHT:-1}"
QSTAR_H_WEIGHT="${QSTAR_H_WEIGHT:-1}"
NNET_BATCH_SIZE="${NNET_BATCH_SIZE:-4096}"

ENV_ORACLE="powderworld_${DIFF}_macro"
ENV_NEURAL="powderworld_${DIFF}_macro_neural"
DATA_DIR="${DCAI_DIR}/deepcubeai/data/powderworld_${DIFF}_macro/search_test"
SEARCH_PKL="${DATA_DIR}/search_test_${SEARCH_TEST_EPS}.pkl"

ORACLE_ENV_MODEL_DIR="${DCAI_DIR}/deepcubeai/saved_env_models/powderworld_${DIFF}_macro_oracle"
ORACLE_HEUR_DIR="${DCAI_DIR}/deepcubeai/saved_heur_models/powderworld_${DIFF}_macro_oracle/current"
ENCODER_ENV_MODEL_DIR="${DCAI_DIR}/deepcubeai/saved_env_models/powderworld_${DIFF}_macro_learned"
ENCODER_HEUR_DIR="${DCAI_DIR}/deepcubeai/saved_heur_models/powderworld_${DIFF}_macro_learned/current"
NEURAL_ENV_MODEL_DIR="${DCAI_DIR}/deepcubeai/saved_env_models/powderworld_${DIFF}_macro_neural"
NEURAL_HEUR_DIR="${DCAI_DIR}/deepcubeai/saved_heur_models/powderworld_${DIFF}_macro_neural/current"

if [[ "${MODE}" == "oracle" ]]; then
  QSTAR_ENV="${ENV_ORACLE}"
  ENV_MODEL_DIR="${ORACLE_ENV_MODEL_DIR}"
  HEUR_DIR="${ORACLE_HEUR_DIR}"
  RESULTS_NAME="${RESULTS_NAME:-qstar_oracle_${SEARCH_TEST_EPS}}"
else
  QSTAR_ENV="${ENV_NEURAL}"
  ENV_MODEL_DIR="${NEURAL_ENV_MODEL_DIR}"
  HEUR_DIR="${NEURAL_HEUR_DIR}"
  RESULTS_NAME="${RESULTS_NAME:-qstar_neural_${SEARCH_TEST_EPS}}"
fi

RESULTS_DIR="${DCAI_DIR}/deepcubeai/results/powderworld_${DIFF}_macro/${RESULTS_NAME}"
REAL_EVAL_JSON="${RESULTS_DIR}/real_env_eval.json"

has_stage() {
  [[ ",${STAGES}," == *",all,"* || ",${STAGES}," == *",$1,"* ]]
}

run_py() {
  echo
  if [[ "${USE_UV}" == "1" ]]; then
    echo "+ uv run python $*"
    if [[ "${DRY_RUN}" != "1" ]]; then
      uv run python "$@"
    fi
  else
    echo "+ python $*"
    if [[ "${DRY_RUN}" != "1" ]]; then
      python "$@"
    fi
  fi
}

echo "DCAI_DIR=${DCAI_DIR}"
echo "MQE_DIR=${MQE_DIR}"
echo "DIFF=${DIFF}"
echo "MODE=${MODE}"
echo "USE_UV=${USE_UV}"
echo "STAGES=${STAGES}"
echo "SEARCH_PKL=${SEARCH_PKL}"
echo "RESULTS_DIR=${RESULTS_DIR}"

if [[ "${USE_UV}" == "1" && "${DRY_RUN}" != "1" ]]; then
  uv run python - <<'PY'
import torch
import ogbench.powderworld  # noqa: F401
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
fi

if [[ "${MODE}" == "oracle" ]] && has_stage "oracle_ckpt"; then
  run_py deepcubeai/scripts/create_powderworld_macro_oracle.py \
    --env "${ENV_ORACLE}" \
    --env_model_dir "${ORACLE_ENV_MODEL_DIR}" \
    --heur_dir "${ORACLE_HEUR_DIR}"
fi

if [[ "${MODE}" == "neural" ]] && has_stage "train_encoder"; then
  run_py deepcubeai/scripts/train_powderworld_macro_encoder.py \
    --difficulty "${DIFF}" \
    --env_model_dir "${ENCODER_ENV_MODEL_DIR}" \
    --heur_dir "${ENCODER_HEUR_DIR}" \
    --num_train_pairs "${NUM_TRAIN_PAIRS}" \
    --num_val_pairs "${NUM_VAL_PAIRS}" \
    --seed "${TRAIN_SEED}" \
    --val_seed "${VAL_SEED}" \
    --iters "${ENCODER_ITRS}" \
    --batch_size "${ENCODER_BATCH}"
fi

if [[ "${MODE}" == "neural" ]] && has_stage "train_models"; then
  run_py deepcubeai/scripts/train_powderworld_macro_models.py \
    --difficulty "${DIFF}" \
    --encoder_checkpoint "${ENCODER_ENV_MODEL_DIR}/encoder_state_dict.pt" \
    --env_model_dir "${NEURAL_ENV_MODEL_DIR}" \
    --heur_dir "${NEURAL_HEUR_DIR}" \
    --iters "${MODEL_ITRS}" \
    --batch_size "${MODEL_BATCH}" \
    --eval_batch_size "${MODEL_EVAL_BATCH}" \
    --eval_batches "${MODEL_EVAL_BATCHES}"
fi

if has_stage "gen_search"; then
  mkdir -p "${DATA_DIR}"
  run_py deepcubeai/scripts/generate_search_test_data.py \
    --env "${ENV_ORACLE}" \
    --num_episodes "${SEARCH_TEST_EPS}" \
    --num_steps -1 \
    --data_file "${SEARCH_PKL}" \
    --start_level "${SEARCH_START_LEVEL}"
fi

if [[ "${FRESH}" == "1" && "${DRY_RUN}" != "1" ]]; then
  rm -f "${RESULTS_DIR}/results.pkl" "${RESULTS_DIR}/output.txt" "${REAL_EVAL_JSON}"
fi

if has_stage "qstar"; then
  mkdir -p "${RESULTS_DIR}"
  run_py deepcubeai/search_methods/qstar_imag.py \
    --env "${QSTAR_ENV}" \
    --states "${SEARCH_PKL}" \
    --env_model "${ENV_MODEL_DIR}" \
    --heur "${HEUR_DIR}" \
    --heur_type dqn \
    --results_dir "${RESULTS_DIR}" \
    --per_eq_tol "${PER_EQ_TOL}" \
    --batch_size "${QSTAR_BATCH}" \
    --weight "${QSTAR_WEIGHT}" \
    --h_weight "${QSTAR_H_WEIGHT}" \
    --nnet_batch_size "${NNET_BATCH_SIZE}" \
    --save_imgs false
fi

if has_stage "real_eval"; then
  run_py deepcubeai/scripts/eval_powderworld_macro_qstar_real.py \
    --difficulty "${DIFF}" \
    --results "${RESULTS_DIR}/results.pkl" \
    --out "${REAL_EVAL_JSON}"
fi

echo
echo "Done."
echo "Search data: ${SEARCH_PKL}"
echo "Q* output: ${RESULTS_DIR}/output.txt"
echo "Q* results: ${RESULTS_DIR}/results.pkl"
echo "Real env eval: ${REAL_EVAL_JSON}"
