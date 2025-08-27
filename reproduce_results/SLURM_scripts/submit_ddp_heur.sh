#!/usr/bin/env bash

# Customize the directives below for your cluster/policy
#SBATCH --job-name=JOB_NAME
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1           # ONE torchrun launcher per node (torchrun will spawn per-GPU workers)
#SBATCH --gres=gpu:2                  # GPUs per node
#SBATCH --cpus-per-task=64            # CPU cores per task
#SBATCH --partition=PARTITION_NAME
#SBATCH --output=/PROJECT/DIRECTORY/DeepCubeAI/reproduce_results/SLURM_scripts/job_run_outputs/%x_%j.out
#SBATCH --error=/PROJECT/DIRECTORY/DeepCubeAI/reproduce_results/SLURM_scripts/job_run_outputs/%x_%j.err
#SBATCH -D /PROJECT/DIRECTORY/DeepCubeAI

##SBATCH --time=2-00:00:00
##SBATCH --constraint=a100|v100
##SBATCH --exclusive
##SBATCH --account=ACCOUNT_NAME
##SBATCH --mail-user=USER@EXAMPLE.COM
##SBATCH --mail-type=END

# # the environment variable PYTHONUNBUFFERED to set unbuffered I/O for the whole batch script
# export PYTHONUNBUFFERED=TRUE

# shellcheck disable=SC1091
. "$HOME/.bash_profile"
# shellcheck disable=SC1091
. "$HOME/.bashrc"

# Function to print system information
print_system_info() {
    if [ -z "${HOSTNAME:-}" ]; then
        HOSTNAME=$(hostname)
    fi
    echo "------------------------------------"
    echo "Configuration Information:"
    echo "------------------------------------"
    echo "Date: $(date +"%m/%d/%Y")"
    echo "Time: $(date +%r)"
    echo "OS: $(uname -s)"
    echo "GPU Model: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | paste -sd ', ')"
    echo "GPU Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | paste -sd ', ')"
    echo "CPU Cores: $(nproc)"
    echo "GPU Cores: $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    echo "Python Version: $(python --version)"
    # Resources
    echo "PWD: $(pwd)"
    echo "Number of tasks (MPI workers) '-n': $(scontrol show job "$SLURM_JOBID" | awk '/NumTasks/ {print $2}' | tr -dc '0-9' | awk '{$1=$1};1')"
    echo "CUDA_VISIBLE_DEVICES (GPUs): $CUDA_VISIBLE_DEVICES"
    echo "Allocated CPU Cores: $SLURM_CPUS_ON_NODE"
    echo "Allocated GPU Cores: $(scontrol show job "$SLURM_JOBID" | grep Gres | awk -F'gpu:' '{print $2}' | awk '{print $1}')"
    echo "HOST: $HOSTNAME"
    echo "Python: $(which python)"
    echo "Partition: $SLURM_PARTITION"
    # Job Information
    echo "Job ID: $SLURM_ID"
    echo "------------------------------------"
    echo "------------------------------------"
}

print_system_info
echo

cd /PROJECT/DIRECTORY/DeepCubeAI || { echo "Directory missing"; exit 1; }

echo "Activating pixi environment..."
pixi install --all
eval "$(pixi shell-hook -e all)"
echo "Python executable: $(which python)"
echo

# Runtime parameters
ENV_NAME="[ENV_NAME]"  # Environment key passed as --env
ENV_MODEL_NAME_DISC="${ENV_NAME}_env_disc" # Directory/name for the discrete environment model. In qlearning_dist.py --env_model expects a dir containing env_state_dict.pt
HEUR_NNET_NAME="${ENV_NAME}_heur_dist" # Heuristic/DQN model name prefix for saves/logs (maps to model_dir, qlearning_dist.py uses --nnet_name)
PER_EQ_TOL=100  # Percent of latent state elements that need to be equal to declare equal
HEUR_BATCH_SIZE=10_000  # Logical GLOBAL batch size per optimizer iteration: per-rank ≈ HEUR_BATCH_SIZE / WORLD_SIZE
STATES_PER_UPDATE=50_000_000  # Total number of states to generate per outer update (global). i.e., How many states to train on before checking if target network should be updated
MAX_SOLVE_STEPS=30  # Solve step cap used in search/target construction
START_STEPS=30  # Maximum number of steps to take from offline states to generate start states
GOAL_STEPS=30  # Maximum number of steps to take from the start states to generate goal states
NUM_TEST=1_000  # Number of validation states used for fixed GBFS evaluation
USE_AMP=1  # Use bf16 mixed precision if supported
USE_COMPILE=1  # Use torch.compile
COMPILE_ALL_MODELS=1  # Compile all models replicas used for data generation, training, and evaluation
LR=0.001  # Initial learning rate
LR_D=0.9999993  # LR decay factor
MAX_ITRS=1_000_000  # Maximum training iterations
UPDATE_NNET_BATCH_SIZE=$HEUR_BATCH_SIZE  # Approximate GLOBAL micro-batch size used to compute gradient accumulation
RING=3  # Ring buffer slots for asynchronous sample generation
SEED=0  # RNG seed


###############################
# Multi-node rendezvous setup #
###############################

# Pattern: srun (one task per node) launches torchrun which spawns per-GPU ranks
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_HOST=${NODES[0]}

# Obtain master IP (handles multi-NIC nodes), fallback to hostname if IP resolution fails
MASTER_ADDR=$(srun -N1 -n1 -w "$MASTER_HOST" hostname --ip-address 2>/dev/null | awk '{print $1}')
if [[ -z "$MASTER_ADDR" || "$MASTER_ADDR" == *":"* ]]; then
    # Some systems return multiple addresses separated by spaces or include IPv6, pick first pure IPv4
    MASTER_ADDR=$(srun -N1 -n1 -w "$MASTER_HOST" hostname -I | awk '{for(i=1;i<=NF;i++){if($i ~ /\./){print $i; exit}}}')
fi
if [[ -z "$MASTER_ADDR" ]]; then
    MASTER_ADDR=$MASTER_HOST
fi
export MASTER_ADDR

BASE_PORT=29500
export MASTER_PORT=${MASTER_PORT:-$((BASE_PORT + (SLURM_JOB_ID % 1000)))}

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l || echo 1)}}
NNODES=${SLURM_JOB_NUM_NODES:-1}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE WORLD_SIZE=$WORLD_SIZE"
echo

# Use InfiniBand/RoCE for NCCL if available (0=enabled/default). Set to 1 only when IB/RoCE is misconfigured or causing hangs/timeouts; otherwise keep 0 or unset so NCCL picks the best transport
# (Disabling forces TCP/IP sockets and can hurt performance.)
# export NCCL_IB_DISABLE=0

# Old var is deprecated. Unset it to avoid conflicting behavior and rely on PyTorch’s replacement below
# (Safe no-op if it wasn’t set.)
# unset NCCL_ASYNC_ERROR_HANDLING 2>/dev/null || true

# PyTorch-side async NCCL error handling policy. 0=no handling, 1=abort NCCL comm + tear down process (fail fast), 2=abort comm only, 3=tear down process only (default)
# Choose 3 (default) unless you explicitly want more aggressive fail-fast behavior (1/2) during debugging/experimentation
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Control when to use GPUDirect RDMA NIC↔GPU. Options: LOC/PIX/PXB/PHB/SYS. Pick based on topology:
#  - PIX (same PCI switch), PXB (across PCI switches), PHB (same NUMA), SYS (cross-NUMA). If unsure, leave unset to let NCCL auto-tune
# export NCCL_NET_GDR_LEVEL=PXB

# Set OpenMP thread count per rank to CPUs Slurm granted to each task. Good default for CPU-heavy dataloading/kernels.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Include C++ stack traces on native errors. Turn on when debugging low-level crashes, keep off in production to reduce log noise and rare edge issues on some platforms
# export TORCH_SHOW_CPP_STACKTRACES=1

# CUDA sync behavior. 0=async (fast, default). Set to 1 only while debugging to get precise error locations, it will slow execution
# export CUDA_LAUNCH_BLOCKING=0

# Tune PyTorch CUDA allocator to reduce fragmentation or handle variable-sized tensors
#  - max_split_size_mb: try 64-512, smaller can reduce fragmentation but may slow allocs
#  - expandable_segments:True can help with dynamic shapes/fragmentation
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True

# Extra torch.distributed logging and desync checks. OFF/INFO/DETAIL. Use DETAIL only when debugging, it can slow training
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# export TORCH_LOGS="+dynamo,inductor,recompiles,guards"

export TORCHINDUCTOR_CACHE_DIR="/work/msoltani/projects/DeepCubeAI/.shared_inductor_cache"

vars=(
    NCCL_IB_DISABLE
    NCCL_ASYNC_ERROR_HANDLING
    TORCH_NCCL_ASYNC_ERROR_HANDLING
    NCCL_NET_GDR_LEVEL
    OMP_NUM_THREADS
    TORCH_SHOW_CPP_STACKTRACES
    CUDA_LAUNCH_BLOCKING
    PYTORCH_CUDA_ALLOC_CONF
    TORCH_DISTRIBUTED_DEBUG
    TORCH_LOGS
    TORCHINDUCTOR_CACHE_DIR
)

for name in "${vars[@]}"; do
    val="${!name-}"               # expand by variable name, empty if unset
    if [[ -n "$val" ]]; then      # only print when non-empty
        printf '%s=%s\n' "$name" "$val"
    fi
done
echo

TORCHRUN_ARGS=(
    --nnodes="$NNODES"
    --nproc-per-node="$GPUS_PER_NODE"
    --rdzv-backend=c10d
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT"
    --rdzv-id="$SLURM_JOB_ID"
    --max-restarts=0
)

TRAIN_ARGS=(
    -m deepcubeai train_heur
    --env "$ENV_NAME"
    --data_dir "$ENV_NAME"
    --data_file_name 10k_stp30
    --env_model_name "$ENV_MODEL_NAME_DISC"
    --heur_nnet_name "$HEUR_NNET_NAME"
    --per_eq_tol "$PER_EQ_TOL"
    --heur_batch_size "$HEUR_BATCH_SIZE"
    --states_per_update "$STATES_PER_UPDATE"
    --start_steps "$START_STEPS"
    --goal_steps "$GOAL_STEPS"
    --max_solve_steps "$MAX_SOLVE_STEPS"
    --num_test "$NUM_TEST"
    --use_dist
    --lr "$LR"
    --lr_d "$LR_D"
    --max_itrs "$MAX_ITRS"
    --update_nnet_batch_size "$UPDATE_NNET_BATCH_SIZE"
    --ring "$RING"
    --seed "$SEED"
)
if (( USE_AMP )); then
    TRAIN_ARGS+=(--amp)
fi
if (( USE_COMPILE )); then
    TRAIN_ARGS+=(--compile)
fi
if (( COMPILE_ALL_MODELS )); then
    TRAIN_ARGS+=(--compile_all_models)
fi

echo "Launching (srun) torchrun ${TORCHRUN_ARGS[*]} ${TRAIN_ARGS[*]}"
echo

# srun (1 task per node) will start torchrun once per node, torchrun spawns GPU workers locally
srun torchrun "${TORCHRUN_ARGS[@]}" "${TRAIN_ARGS[@]}"
