#!/bin/sh
#SBATCH --chdir=/PROJECT/DIRECTORY/DeepCubeAI  # Sets the working directory for the job
#SBATCH --job-name=JOB_NAME
#SBATCH --output=/PROJECT/DIRECTORY/DeepCubeAI/reproduce_results/SLURM_scripts/job_run_outputs/%x_%j.out  # Standard output file (%x: job name, %j: job ID)
#SBATCH --error=/PROJECT/DIRECTORY/DeepCubeAI/reproduce_results/SLURM_scripts/job_run_outputs/%x_%j.err  # Standard error file (%x: job name, %j: job ID)
#SBATCH -N 1
#SBATCH -p PARTITION_NAME
#SBATCH --gres=gpu:2  # Number of GPUs
#SBATCH --cpus-per-task=64  # Number of CPU cores per task

## SBATCH --constraint=a100|v100  # Specifies GPU type constraints. Modify based on availability
## SBATCH --exclusive  # Allocate all resources on the node
## SBATCH --nodes=1  # Number of nodes to allocate
## SBATCH --ntasks=1  # Total number of tasks across all nodes
## SBATCH --mem=16G  # Total memory allocation for the job
## SBATCH --nodelist=node
## SBATCH --time=1-00:00:00  # The time format is days-hours:minutes:seconds
## SBATCH --account ACCOUNT_NAME

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
echo ""

cmd="cd /PROJECT/DIRECTORY/DeepCubeAI || exit"
printf "Running the command '%s'...\n" "$cmd"
$cmd

# echo "Running the command 'pixi update'..."
# pixi update

cmd="pixi install --all"
printf "Running the command '%s'...\n" "$cmd"
$cmd
echo ""

echo "Running the command for the shell hook (using the env 'all' )..."
eval "$(pixi shell-hook -e all)"
echo ""

echo "Python executable: $(which python)"
echo; echo

cmd="bash reproduce_results/run_directly/reproduce_[ENV_NAME].sh" # e.g. reproduce_cube3.sh
printf "Running the command '%s'...\n" "$cmd"
$cmd
echo ""
