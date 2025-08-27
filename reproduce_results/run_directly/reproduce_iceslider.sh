#!/usr/bin/env bash

set -euo pipefail
# Script path is .../DeepCubeAI/reproduce_results/run_directly/reproduce_icelsider.sh
# Go up three levels to reach .../DeepCubeAI
DCAI_DIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
cd ..
cd "$DCAI_DIR" || exit 1
echo "Working directory: $(pwd)"

run_pipeline() {
    local CMD="$1"
    echo "Running command:"
    # Use shell parameter expansion instead of sed (SC2001)
    local PRETTY_CMD="${CMD// --/$'\n--'}"
    while IFS= read -r line; do
        echo "$line"
    done <<< "$PRETTY_CMD"
    echo ""
    echo "------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------"
    $CMD
    echo "------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------"
    echo ""
}

ENV=iceslider
DATA_DIR=iceslider
ENV_MODEL_NAME_DISC=iceslider_disc
ENV_MODEL_NAME_CONT=iceslider_cont
ENV_MODEL_DIR_DISC=deepcubeai/saved_env_models/${ENV_MODEL_NAME_DISC}
ENV_MODEL_DIR_CONT=deepcubeai/saved_env_models/${ENV_MODEL_NAME_CONT}
HEUR_NNET_NAME=iceslider_heur
DATA_FILE_NAME_TRAIN_VAL=s0-1k_stp20
DATA_FILE_NAME_MODEL_TEST=s5k-5.1k_stp1k
DATA_FILE_NAME_MODEL_TEST_PLOT=s5k-5.1k_stp10k
DATA_FILE_NAME_SEARCH_TEST=s2k-2.1k
QSTAR_WEIGHT=0.7
QSTAR_H_WEIGHT=1.0
QSTAR_BATCH_SIZE=1
UCS_BATCH_SIZE=50
current_time=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR_QSTAR="model=${ENV_MODEL_NAME_DISC}__heur=${HEUR_NNET_NAME}_QSTAR_results/path_cost_weight=${QSTAR_WEIGHT}__h_weight=${QSTAR_H_WEIGHT}__batchsize=${QSTAR_BATCH_SIZE}_${current_time}"
RESULTS_DIR_UCS="model=${ENV_MODEL_NAME_DISC}_UCS_results/batchsize=${UCS_BATCH_SIZE}_${current_time}"
RESULTS_DIR_GBFS="model=${ENV_MODEL_NAME_DISC}__heur=${HEUR_NNET_NAME}_GBFS_results/${current_time}"
PER_EQ_TOL=100
PLOTS_SAVE_DIR="${DCAI_DIR}/deepcubeai/"
# Get the number of CPU cores available on the system
# Resolve the directory of this script, then source the sibling file.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=get_cpu_num.sh
. "$SCRIPT_DIR/get_cpu_num.sh"

CORES="$(get_allocated_cpus)"
if (( CORES > 1 )); then
  NUM_CORES=$((CORES - 1))
else
  NUM_CORES=1
fi

CMD_TRAIN_VAL="python -m deepcubeai gen_offline \
                                    --env $ENV \
                                    --data_dir $DATA_DIR \
                                    --data_file_name $DATA_FILE_NAME_TRAIN_VAL \
                                    --num_offline_steps 20 \
                                    --num_train_eps 20000 \
                                    --num_val_eps 5000 \
                                    --num_cpus $NUM_CORES \
                                    --start_level 0 \
                                    --num_levels 1000"

CMD_ENV_MODEL_TEST="python -m deepcubeai gen_env_test \
                                         --env $ENV \
                                         --data_dir $DATA_DIR \
                                         --data_file_name $DATA_FILE_NAME_MODEL_TEST \
                                         --num_offline_steps 1000 \
                                         --num_test_eps 100 \
                                         --num_cpus $NUM_CORES \
                                         --start_level 5000 \
                                         --num_levels 100"

CMD_SEARCH_TEST="python -m deepcubeai gen_search_test \
                                      --env $ENV \
                                      --data_dir $DATA_DIR \
                                      --data_file_name $DATA_FILE_NAME_SEARCH_TEST \
                                      --num_test_eps 100 \
                                      --start_level 2000"

CMD_TRAIN_ENV_DISC="python -m deepcubeai train_model_disc \
                                         --env $ENV \
                                         --data_dir $DATA_DIR \
                                         --data_file_name $DATA_FILE_NAME_TRAIN_VAL \
                                         --env_batch_size 100 \
                                         --env_model_name $ENV_MODEL_NAME_DISC"

CMD_TEST_ENV_DISC="python -m deepcubeai test_model \
                                        --env $ENV \
                                        --data_dir $DATA_DIR \
                                        --data_file_name $DATA_FILE_NAME_MODEL_TEST \
                                        --env_model_name $ENV_MODEL_NAME_DISC \
                                        --print_interval 50"

CMD_TRAIN_ENV_CONT="python -m deepcubeai train_model_cont \
                                         --env $ENV \
                                         --data_dir $DATA_DIR \
                                         --data_file_name $DATA_FILE_NAME_TRAIN_VAL \
                                         --env_batch_size 100 \
                                         --env_model_name $ENV_MODEL_NAME_CONT"

CMD_TEST_ENV_CONT="python -m deepcubeai test_model_cont \
                                        --env $ENV \
                                        --data_dir $DATA_DIR \
                                        --data_file_name $DATA_FILE_NAME_MODEL_TEST \
                                        --env_model_name $ENV_MODEL_NAME_CONT \
                                        --print_interval 50"

CMD_ENCODE_OFFLINE="python -m deepcubeai encode_offline \
                                         --env $ENV \
                                         --data_dir $DATA_DIR \
                                         --data_file_name $DATA_FILE_NAME_TRAIN_VAL \
                                         --env_model_name $ENV_MODEL_NAME_DISC"

CMD_TRAIN_HEUR="python -m deepcubeai train_heur \
                                     --env $ENV \
                                     --data_dir $DATA_DIR \
                                     --data_file_name $DATA_FILE_NAME_TRAIN_VAL \
                                     --env_model_name $ENV_MODEL_NAME_DISC \
                                     --heur_nnet_name $HEUR_NNET_NAME \
                                     --per_eq_tol $PER_EQ_TOL \
                                     --heur_batch_size 10_000 \
                                     --states_per_update 50_000_000 \
                                     --start_steps 20 \
                                     --goal_steps 20 \
                                     --max_solve_steps 20 \
                                     --num_test 1000"

CMD_QSTAR="python -m deepcubeai qstar \
                                --env $ENV \
                                --data_dir $DATA_DIR \
                                --data_file_name $DATA_FILE_NAME_SEARCH_TEST \
                                --env_model_name $ENV_MODEL_NAME_DISC \
                                --heur_nnet_name $HEUR_NNET_NAME \
                                --qstar_batch_size $QSTAR_BATCH_SIZE \
                                --qstar_weight $QSTAR_WEIGHT \
                                --qstar_h_weight $QSTAR_H_WEIGHT \
                                --per_eq_tol $PER_EQ_TOL \
                                --qstar_results_dir $RESULTS_DIR_QSTAR \
                                --no_save_imgs"

CMD_UCS="python -m deepcubeai ucs \
                              --env $ENV \
                              --data_dir $DATA_DIR \
                              --data_file_name $DATA_FILE_NAME_SEARCH_TEST \
                              --env_model_name $ENV_MODEL_NAME_DISC \
                              --ucs_batch_size $UCS_BATCH_SIZE \
                              --per_eq_tol $PER_EQ_TOL \
                              --ucs_results_dir $RESULTS_DIR_UCS \
                              --no_save_imgs"

CMD_GBFS="python -m deepcubeai gbfs \
                               --env $ENV \
                               --data_dir $DATA_DIR \
                               --data_file_name $DATA_FILE_NAME_SEARCH_TEST \
                               --env_model_name $ENV_MODEL_NAME_DISC \
                               --heur_nnet_name $HEUR_NNET_NAME \
                               --per_eq_tol $PER_EQ_TOL \
                               --gbfs_results_dir $RESULTS_DIR_GBFS \
                               --search_itrs 100"

CMD_VIZ_DATA="python -m deepcubeai visualize_data \
                                   --env $ENV \
                                   --data_dir $DATA_DIR \
                                   --data_file_name $DATA_FILE_NAME_TRAIN_VAL \
                                   --num_train_trajs_viz 8 \
                                   --num_train_steps_viz 2 \
                                   --num_val_trajs_viz 8 \
                                   --num_val_steps_viz 2"

CMD_ENV_MODEL_TEST_PLOT="python -m deepcubeai gen_env_test \
                                              --env $ENV \
                                              --data_dir $DATA_DIR \
                                              --data_file_name $DATA_FILE_NAME_MODEL_TEST_PLOT \
                                              --num_offline_steps 10_000 \
                                              --num_test_eps 100 \
                                              --num_cpus $NUM_CORES \
                                              --start_level 5000 \
                                              --num_levels 100"

CMD_DISC_VS_CONT="python -m deepcubeai disc_vs_cont \
                                       --env $ENV \
                                       --data_dir $DATA_DIR \
                                       --data_file_name $DATA_FILE_NAME_MODEL_TEST_PLOT \
                                       --env_model_dir_disc $ENV_MODEL_DIR_DISC \
                                       --env_model_dir_cont $ENV_MODEL_DIR_CONT \
                                       --save_dir $PLOTS_SAVE_DIR \
                                       --num_steps 10_000 \
                                       --num_episodes 100 \
                                       --print_interval 500"

# gen_offline
run_pipeline "$CMD_TRAIN_VAL"

# visualize_data
run_pipeline "$CMD_VIZ_DATA"

# gen_offline_test
run_pipeline "$CMD_ENV_MODEL_TEST"

# gen_offline_test (10K steps for plotting)
run_pipeline "$CMD_ENV_MODEL_TEST_PLOT"

# gen_search_test
run_pipeline "$CMD_SEARCH_TEST"

# train_model_disc
run_pipeline "$CMD_TRAIN_ENV_DISC"

# test_model
run_pipeline "$CMD_TEST_ENV_DISC"

# train_model_cont
run_pipeline "$CMD_TRAIN_ENV_CONT"

# test_model_cont
run_pipeline "$CMD_TEST_ENV_CONT"

# disc_vs_cont
run_pipeline "$CMD_DISC_VS_CONT"

# encode_offline
run_pipeline "$CMD_ENCODE_OFFLINE"

# train_heur
run_pipeline "$CMD_TRAIN_HEUR"

# gbfs
run_pipeline "$CMD_GBFS"

# qstar
run_pipeline "$CMD_QSTAR"

# ucs
run_pipeline "$CMD_UCS"
