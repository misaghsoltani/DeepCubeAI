#!/bin/sh
#SBATCH --job-name=cube3_dist_heur
#SBATCH -N 10
#SBATCH -D /project/dir/
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH --output=job_run_outputs/cube3_dist_heur_job%j.out
#SBATCH --error=job_run_outputs/cube3_dist_heur_job%j.err
#SBATCH -p OOD_gpu_32gb,gpu-v100-32gb

# #SBATCH --mail-user=msoltani@email.sc.edu
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
    echo "CPU Cores: $(nproc)"
    echo "GPU Cores: $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
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
    echo "CUDA_VISIBLE_DEVICES (GPUs): $CUDA_VISIBLE_DEVICES"
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

    setup_mpi
    # Run the pipeline script
    mpirun -np $NP_OPTION \
           -H $H_OPTION \
           -x MASTER_ADDR=$MASTER_ADDR \
           -x MASTER_PORT=$MASTER_PORT \
           -x PATH \
           -bind-to none -map-by slot \
           -mca pml ob1 -mca btl ^openib \
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

    # Display elapsed time in the desired format
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
current_time=$(date +"%Y%m%d_%H%M%S%3N")
HEUR_NNET_NAME=cube3_heur_dist
DATA_FILE_NAME_TRAIN_VAL=s0-1k_stp20
PER_EQ_TOL=100


CMD_TRAIN_HEUR="bash scripts/pipeline.sh --stage train_heur \
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
                                         --max_solve_steps 30
                                         --use_dist"


# train_heur
run_pipeline "$CMD_TRAIN_HEUR"