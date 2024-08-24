#!/bin/sh
#SBATCH --job-name=cube3
#SBATCH -N 1
#SBATCH -D /project/dir/deepcubeai/
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --output=job_run_outputs/cube3_job%j.out
#SBATCH --error=job_run_outputs/cube3_job%j.err
#SBATCH -p partition_name

# #SBATCH --mail-user=username@email.com
# #SBATCH --mail-type=END
# #SBATCH --exclusive  # Allocate all resources on the node

# the environment variable PYTHONUNBUFFERED to set unbuffered I/O for the whole batch script
export PYTHONUNBUFFERED=TRUE

# Load modules
# module load python3/anaconda/2023.9
source ~/.bash_profile
conda activate DeepCubeAI_env
cd /project/dir/
#module load cuda/12.3
source /project/dir/setup.sh


# Function to print system information
print_system_info() {
    echo "------------------------------------------------------------------------"
    echo "Configuration Information:"
    echo "------------------------------------------------------------------------"
    echo "Date: $(date +"%m/%d/%Y")"
    echo "Time: $(date +%r)"
    echo "OS: $(uname -s)"
    echo "Kernel: $(uname -r)"
    echo "Memory: $(free -m | grep Mem | awk '{print $2}') MB"
    echo "GPU Model: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | paste -sd ', ')"
    echo "GPU Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | paste -sd ', ')"
    echo "CPU Cores Avail on Machine: $(nproc)"
    echo "GPU Cores Avail on Machine: $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    echo "CUDA Version: $(nvcc --version | awk '/release/ {print $6, $7, $8}')"
    echo "Python Version: $(python --version)"
    echo "NumPy Version: $(python -c 'import numpy as np; print(np.__version__)')"
    echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
    echo "Matplotlib Version: $(python -c 'import matplotlib; print(matplotlib.__version__)')"
    # Project Information
    echo "Project Directory: $(git rev-parse --show-toplevel)"
    # commit=$(git rev-parse HEAD 2>/dev/null)
    # echo "Git Commit: ${commit:-N/A}"
    # Resources
    echo "PWD: $(pwd)"
    echo "Number of tasks (MPI workers) '-n': $(scontrol show job $SLURM_JOBID | awk '/NumTasks/ {print $2}' | tr -dc '0-9' | awk '{$1=$1};1')"
    echo "CUDA_VISIBLE_DEVICES (GPU numbers): $CUDA_VISIBLE_DEVICES"
    echo "Allocated CPU Cores: $SLURM_CPUS_ON_NODE"
    echo "Allocated GPU Cores: $(scontrol show job $SLURM_JOBID | grep Gres | awk -F'gpu:' '{print $2}' | awk '{print $1}')"
    echo "HOST: $HOSTNAME"
    echo "Python: $(which python)"
    echo "CONDA: $CONDA_PREFIX"
    echo "Partition: $SLURM_JOB_PARTITION"
    # Job Information
    echo "Job ID: $SLURM_JOB_ID"
    # User Info
    echo "User: $(whoami)"
    # Check to see if GPU is available
    echo "GPU Status: $(python3 -c 'import torch; print("Running on GPU (CUDA is available)." if torch.cuda.is_available() else "NOT running on GPU (CUDA is not available).")')"
    echo ""
    echo "------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------"
    echo ""
}

# Print system information
print_system_info

run_pipeline() {

    local CMD=$1

    echo "Running command:"
    while IFS= read -r line; do
        echo "$line"
    done <<< "$(echo "$CMD" | sed 's/ --/\n--/g')"
    echo ""

    echo "------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------"

    # Capture start time
    START_TIME=$(date +%s%3N)

    # Run the pipeline script
    $CMD

    # Capture end time
    END_TIME=$(date +%s%3N)

    # Calculate execution time in milliseconds
    ELAPSED_TIME=$((END_TIME - START_TIME))

    # Convert milliseconds to days, hours, minutes, seconds, and milliseconds
    DAYS=$((ELAPSED_TIME / 86400000))
    ELAPSED_TIME=$((ELAPSED_TIME % 86400000))

    HOURS=$((ELAPSED_TIME / 3600000))
    ELAPSED_TIME=$((ELAPSED_TIME % 3600000))

    MINUTES=$((ELAPSED_TIME / 60000))
    ELAPSED_TIME=$((ELAPSED_TIME % 60000))

    SECONDS=$((ELAPSED_TIME / 1000))
    MILLISECONDS=$((ELAPSED_TIME % 1000))

    echo "------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------"
    echo "Elapsed Time for this stage (D:H:M:S:MS): $DAYS:$HOURS:$MINUTES:$SECONDS:$MILLISECONDS"
    echo "------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------"
    echo ""
}


ENV=cube3
DATA_DIR=cube3
ENV_MODEL_NAME_DISC=cube3_disc
ENV_MODEL_NAME_CONT=cube3_cont
ENV_MODEL_DIR_DISC=deepcubeai/saved_env_models/${ENV_MODEL_NAME_DISC}
ENV_MODEL_DIR_CONT=deepcubeai/saved_env_models/${ENV_MODEL_NAME_CONT}
HEUR_NNET_NAME=cube3_heur
DATA_FILE_NAME_TRAIN_VAL=10k_stp30
DATA_FILE_NAME_MODEL_TEST=0.1k_stp1k
DATA_FILE_NAME_MODEL_TEST_PLOT=0.1k_stp10k
SEARCH_TEST_DATA=deepcubeai/data/cube3/search_test/search_test_data.pkl
SEARCH_TEST_DATA_REVERSE=deepcubeai/data/cube3/search_test/search_test_data_reverse.pkl
QSTAR_WEIGHT=0.6
QSTAR_H_WEIGHT=1.0
QSTAR_BATCH_SIZE=10_000
UCS_BATCH_SIZE=10_000
current_time=$(date +"%Y%m%d_%H%M%S")$(($(date +%N)/1000000))
RESULTS_DIR_QSTAR="model=${ENV_MODEL_NAME_DISC}__heur=${HEUR_NNET_NAME}__QSTAR_results/path_cost_weight=${QSTAR_WEIGHT}__h_weight=${QSTAR_H_WEIGHT}__batchsize=${QSTAR_BATCH_SIZE}_${current_time}"
RESULTS_DIR_UCS="model=${ENV_MODEL_NAME_DISC}__UCS_results/batchsize=${UCS_BATCH_SIZE}_${current_time}"
RESULTS_DIR_GBFS="model=${ENV_MODEL_NAME_DISC}__heur=${HEUR_NNET_NAME}__GBFS_results/${current_time}"
PER_EQ_TOL=100
PLOTS_SAVE_DIR="${DCAI_DIR}/deepcubeai/"



CMD_TRAIN_VAL="bash deepcubeai.sh --stage gen_offline \
                                  --env $ENV \
                                  --data_dir $DATA_DIR \
                                  --data_file_name $DATA_FILE_NAME_TRAIN_VAL \
                                  --num_offline_steps 30 \
                                  --num_train_eps 9000 \
                                  --num_val_eps 1000 \
                                  --num_cpus $SLURM_CPUS_ON_NODE"

CMD_ENV_MODEL_TEST="bash deepcubeai.sh --stage gen_env_test \
                                       --env $ENV \
                                       --data_dir $DATA_DIR \
                                       --data_file_name $DATA_FILE_NAME_MODEL_TEST \
                                       --num_offline_steps 1000 \
                                       --num_test_eps 100 \
                                       --num_cpus $SLURM_CPUS_ON_NODE"

CMD_TRAIN_ENV_DISC="bash deepcubeai.sh --stage train_model \
                                       --env $ENV \
                                       --data_dir $DATA_DIR \
                                       --data_file_name $DATA_FILE_NAME_TRAIN_VAL \
                                       --env_batch_size 100 \
                                       --env_model_name $ENV_MODEL_NAME_DISC"

CMD_TEST_ENV_DISC="bash deepcubeai.sh --stage test_model \
                                      --env $ENV \
                                      --data_dir $DATA_DIR \
                                      --data_file_name $DATA_FILE_NAME_MODEL_TEST \
                                      --env_model_name $ENV_MODEL_NAME_DISC \
                                      --print_interval 100"

CMD_TRAIN_ENV_CONT="bash deepcubeai.sh --stage train_model_cont \
                                       --env $ENV \
                                       --data_dir $DATA_DIR \
                                       --data_file_name $DATA_FILE_NAME_TRAIN_VAL \
                                       --env_batch_size 100 \
                                       --env_model_name $ENV_MODEL_NAME_CONT"

CMD_TEST_ENV_CONT="bash deepcubeai.sh --stage test_model_cont \
                                      --env $ENV \
                                      --data_dir $DATA_DIR \
                                      --data_file_name $DATA_FILE_NAME_MODEL_TEST \
                                      --env_model_name $ENV_MODEL_NAME_CONT \
                                      --print_interval 100"

CMD_ENCODE_OFFLINE="bash deepcubeai.sh --stage encode_offline \
                                       --env $ENV \
                                       --data_dir $DATA_DIR \
                                       --data_file_name $DATA_FILE_NAME_TRAIN_VAL \
                                       --env_model_name $ENV_MODEL_NAME_DISC"

CMD_TRAIN_HEUR="bash deepcubeai.sh --stage train_heur \
                                   --env $ENV \
                                   --data_dir $DATA_DIR \
                                   --data_file_name $DATA_FILE_NAME_TRAIN_VAL \
                                   --env_model_name $ENV_MODEL_NAME_DISC \
                                   --heur_nnet_name $HEUR_NNET_NAME \
                                   --per_eq_tol $PER_EQ_TOL \
                                   --heur_batch_size 10_000 \
                                   --states_per_update 50_000_000 \
                                   --start_steps 30 \
                                   --goal_steps 30 \
                                   --max_solve_steps 30"

CMD_QSTAR="bash deepcubeai.sh --stage qstar \
                              --env $ENV \
                              --env_model_name $ENV_MODEL_NAME_DISC \
                              --heur_nnet_name $HEUR_NNET_NAME \
                              --qstar_batch_size $QSTAR_BATCH_SIZE \
                              --qstar_weight $QSTAR_WEIGHT \
                              --qstar_h_weight $QSTAR_H_WEIGHT \
                              --per_eq_tol $PER_EQ_TOL \
                              --qstar_results_dir $RESULTS_DIR_QSTAR \
                              --search_test_data $SEARCH_TEST_DATA \
                              --save_imgs true"

CMD_UCS="bash deepcubeai.sh --stage ucs \
                            --env $ENV \
                            --env_model_name $ENV_MODEL_NAME_DISC \
                            --ucs_batch_size $UCS_BATCH_SIZE \
                            --per_eq_tol $PER_EQ_TOL \
                            --ucs_results_dir $RESULTS_DIR_UCS \
                            --search_test_data $SEARCH_TEST_DATA \
                            --save_imgs true"

CMD_GBFS="bash deepcubeai.sh --stage gbfs \
                             --env $ENV \
                             --env_model_name $ENV_MODEL_NAME_DISC \
                             --heur_nnet_name $HEUR_NNET_NAME \
                             --per_eq_tol $PER_EQ_TOL \
                             --gbfs_results_dir $RESULTS_DIR_GBFS \
                             --search_test_data $SEARCH_TEST_DATA \
                             --search_itrs 100"

CMD_QSTAR_REVERSE_DATA="bash deepcubeai.sh --stage qstar \
                                           --env $ENV \
                                           --env_model_name $ENV_MODEL_NAME_DISC \
                                           --heur_nnet_name $HEUR_NNET_NAME \
                                           --qstar_batch_size $QSTAR_BATCH_SIZE \
                                           --qstar_weight $QSTAR_WEIGHT \
                                           --qstar_h_weight $QSTAR_H_WEIGHT \
                                           --per_eq_tol $PER_EQ_TOL \
                                           --qstar_results_dir $RESULTS_DIR_QSTAR \
                                           --search_test_data $SEARCH_TEST_DATA_REVERSE \
                                           --save_imgs true"

CMD_GBFS_REVERSE_DATA="bash deepcubeai.sh --stage gbfs \
                                          --env $ENV \
                                          --env_model_name $ENV_MODEL_NAME_DISC \
                                          --heur_nnet_name $HEUR_NNET_NAME \
                                          --per_eq_tol $PER_EQ_TOL \
                                          --gbfs_results_dir $RESULTS_DIR_GBFS \
                                          --search_test_data $SEARCH_TEST_DATA_REVERSE \
                                          --search_itrs 100"

CMD_VIZ_DATA="bash deepcubeai.sh --stage visualize_data \
                                 --env $ENV \
                                 --data_dir $DATA_DIR \
                                 --data_file_name $DATA_FILE_NAME_TRAIN_VAL \
                                 --num_train_trajs_viz 8 \
                                 --num_train_steps_viz 2 \
                                 --num_val_trajs_viz 8 \
                                 --num_val_steps_viz 2"

CMD_ENV_MODEL_TEST_PLOT="bash deepcubeai.sh --stage gen_env_test \
                                            --env $ENV \
                                            --data_dir $DATA_DIR \
                                            --data_file_name $DATA_FILE_NAME_MODEL_TEST_PLOT \
                                            --num_offline_steps 10_000 \
                                            --num_test_eps 100 \
                                            --num_cpus $SLURM_CPUS_ON_NODE"

CMD_DISC_VS_CONT="bash deepcubeai.sh --stage disc_vs_cont \
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

# train_model
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

# qstar
run_pipeline "$CMD_QSTAR"

# # ucs
# run_pipeline "$CMD_UCS"

# gbfs
run_pipeline "$CMD_GBFS"

# qstar (reverse data)
run_pipeline "$CMD_QSTAR_REVERSE_DATA"

# gbfs (reverse data)
run_pipeline "$CMD_GBFS_REVERSE_DATA"