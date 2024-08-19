# Display help message
show_help() {
    echo "Usage: pipeline.sh [options]"
    echo ""
    echo "This script executes the pipeline for DeepCubeAI based on the provided arguments."
    echo "Ensure you have set the necessary arguments before running this pipeline."
    echo ""
    echo "Arguments:"
    echo "  --stage                The stage of the pipeline to execute. Available stages are gen_offline, train_model, train_model_cont, test_model, test_model_cont,"
    echo "                         train_heur, qstar, and visualize_data."
    echo "  --env                  The environment for DeepCubeAI."
    echo "  --data_dir             Directory to offline data."
    echo "  --num_offline_steps    Number of steps to be taken for generating offline data."
    echo "  --env_model_name       The environment model file."
    echo "  --batch_size           Batch size."
    echo "  --states_per_update    Number of states per update."
    echo "  --max_solve_steps      Number of steps to take when trying to solve training states with greedy best-first search (GBFS)."
    echo "                         Each state encountered when solving is added to the training set. Number of steps starts at 1"
    echo "                         and is increased every update until the maximum number is reached. Increasing this number can make"
    echo "                         the cost-to-go function more robust by exploring more of the state space."
    echo "  --start_steps          Maximum number of steps to take from offline states to generate start states"
    echo "  --goal_steps           Maximum number of steps to take from the start states to generate goal states"
    echo "  --num_cpus             Number of CPUs to use."
    echo "  --search_test_data     Test data directory."
    echo "  --qstar_batch_size     Batch size for Q* search."
    echo "  --qstar_weight         Weight for heuristic in Q* search."
    echo "  --qstar_results_dir    Directory to save Q* search results."
    echo "  --per_eq_tol           Percent of latent state elements that need to be equal to declare equal."
    echo "  --num_train_eps        Number of episodes for training in offline data generation. Default is 9000."
    echo "  --num_val_eps          Number of episodes for validation in offline data generation. Default is 1000."
    echo "  --num_train_trajs_viz  Number of training trajectories to be visualized as samples. Default is 30."
    echo "  --num_train_steps_viz  Number of training steps per trajectory to be visualized as samples. Default is 10."
    echo "  --num_val_trajs_viz    Number of validation trajectories to be visualized as samples. Default is 30."
    echo "  --num_val_steps_viz    Number of validation steps per trajectory to be visualized as samples. Default is 10."
    echo "  --start_level          Starting seed of offline data. Default is None."
    echo "  --num_levels           Number of unique seeds used for generating offline data. Default is None."
    echo ""
    exit 0
}

if [[ "$1" == "--help" ]]; then
    show_help
fi

# Remove all environment variables that start with 'DCAI_',
# to avoid any potential conflicts or leftovers
unset_vars() {
    unset ${!DCAI_*}
}
unset_vars

while [[ $# -gt 0 ]]; do
    case "$1" in
    --stage)
        DCAI_STAGE="$2"
        shift 2
        ;;
    --env)
        DCAI_ENV="$2"
        shift 2
        ;;
    --num_offline_steps)
        DCAI_NUM_OFFLINE_STEPS="$2"
        shift 2
        ;;
    --start_level)
        DCAI_START_LEVEL="$2"
        shift 2
        ;;
    --num_levels)
        DCAI_NUM_LEVELS="$2"
        shift 2
        ;;
    --env_model_name)
        DCAI_ENV_MODEL_NAME="$2"
        shift 2
        ;;
    --heur_nnet_name)
        DCAI_HEUR_NNET_NAME="$2"
        shift 2
        ;;
    --heur_batch_size)
        DCAI_HEUR_BATCH_SIZE="$2"
        shift 2
        ;;
    --states_per_update)
        DCAI_STATES_PER_UPDATE="$2"
        shift 2
        ;;
    --max_solve_steps)
        DCAI_MAX_SOLVE_STEPS="$2"
        shift 2
        ;;
    --start_steps)
        DCAI_START_STEPS="$2"
        shift 2
        ;;
    --goal_steps)
        DCAI_GOAL_STEPS="$2"
        shift 2
        ;;
    --num_cpus)
        DCAI_NUM_CPUS="$2"
        shift 2
        ;;
    --search_test_data)
        DCAI_SEARCH_TEST_DATA="$2"
        shift 2
        ;;
    --qstar_batch_size)
        DCAI_QSTAR_BATCH_SIZE="$2"
        shift 2
        ;;
    --ucs_batch_size)
        DCAI_UCS_BATCH_SIZE="$2"
        shift 2
        ;;
    --qstar_weight)
        DCAI_QSTAR_WEIGHT="$2"
        shift 2
        ;;
    --qstar_h_weight)
        DCAI_QSTAR_H_WEIGHT="$2"
        shift 2
        ;;
    --qstar_results_dir)
        DCAI_QSTAR_RESULTS_DIR="$2"
        shift 2
        ;;
    --ucs_results_dir)
        DCAI_UCS_RESULTS_DIR="$2"
        shift 2
        ;;
    --gbfs_results_dir)
        DCAI_GBFS_RESULTS_DIR="$2"
        shift 2
        ;;
    --search_itrs)
        DCAI_SEARCH_ITRS="$2"
        shift 2
        ;;
    --per_eq_tol)
        DCAI_PER_EQ_TOL="$2"
        shift 2
        ;;
    --data_dir)
        DCAI_DATA_DIR="$2"
        shift 2
        ;;
    --data_file_name)
        DCAI_DATA_FILE_NAME="$2"
        shift 2
        ;;
    --num_train_eps)
        DCAI_NUM_OFFLINE_EPS_TRAIN="$2"
        shift 2
        ;;
    --num_val_eps)
        DCAI_NUM_OFFLINE_EPS_VAL="$2"
        shift 2
        ;;
    --num_test_eps)
        DCAI_NUM_OFFLINE_EPS_TEST="$2"
        shift 2
        ;;
    --num_train_trajs_viz)
        DCAI_NUM_TRAIN_TRAJS_VIZ="$2"
        shift 2
        ;;
    --num_train_steps_viz)
        DCAI_NUM_TRAIN_STEPS_VIZ="$2"
        shift 2
        ;;
    --num_val_trajs_viz)
        DCAI_NUM_VAL_TRAJS_VIZ="$2"
        shift 2
        ;;
    --num_val_steps_viz)
        DCAI_NUM_VAL_STEPS_VIZ="$2"
        shift 2
        ;;
    --model_test_data_dir)
        DCAI_MODEL_TEST_DATA="$2"
        shift 2
        ;;
    --env_model_dir_disc)
        DCAI_ENV_MODEL_DIR_DISC="$2"
        shift 2
        ;;
    --env_model_dir_cont)
        DCAI_ENV_MODEL_DIR_CONT="$2"
        shift 2
        ;;
    --num_episodes)
        DCAI_NUM_EPISODES="$2"
        shift 2
        ;;
    --num_steps)
        DCAI_NUM_STEPS="$2"
        shift 2
        ;;
    --save_dir)
        DCAI_SAVE_DIR="$2"
        shift 2
        ;;
    --env_batch_size)
        DCAI_ENV_TRAIN_BATCH_SIZE="$2"
        shift 2
        ;;
    --print_interval)
        DCAI_PRINT_INTERVAL="$2"
        shift 2
        ;;
    --save_imgs)
        DCAI_SAVE_IMGS="$2"
        shift 2
        ;;
    --num_test)
        DCAI_NUM_TEST="$2"
        shift 2
        ;;
    --use_dist)
        DCAI_USE_DIST=true
        shift 1
        ;;
    *)
        echo ""
        echo "ARG ERROR: Unknown option: $1"
        echo ""
        unset_vars
        exit 1
        ;;
    esac
done

handle_offline_data_args() {
    # Check if both DCAI_NUM_OFFLINE_EPS_TRAIN and DCAI_NUM_OFFLINE_EPS_VAL are not provided
    if [ -z "$DCAI_NUM_OFFLINE_EPS_TRAIN" ] && [ -z "$DCAI_NUM_OFFLINE_EPS_VAL" ]; then
        # both of them will be the default value
        DCAI_NUM_OFFLINE_EPS_TRAIN_DEFAULT=9000
        DCAI_NUM_OFFLINE_EPS_VAL_DEFAULT=1000
        DCAI_NUM_OFFLINE_EPS_TRAIN=$DCAI_NUM_OFFLINE_EPS_TRAIN_DEFAULT
        DCAI_NUM_OFFLINE_EPS_VAL=$DCAI_NUM_OFFLINE_EPS_VAL_DEFAULT
    fi

    # If only DCAI_NUM_OFFLINE_EPS_TRAIN is provided, calculate DCAI_NUM_OFFLINE_EPS_VAL
    if [ -n "$DCAI_NUM_OFFLINE_EPS_TRAIN" ] && [ -z "$DCAI_NUM_OFFLINE_EPS_VAL" ]; then
        # DCAI_NUM_OFFLINE_EPS_VAL should be %10 of the whole data
        DCAI_NUM_OFFLINE_EPS_VAL=$((DCAI_NUM_OFFLINE_EPS_TRAIN / 9))
    fi

    # If only DCAI_NUM_OFFLINE_EPS_VAL is provided, calculate DCAI_NUM_OFFLINE_EPS_TRAIN
    if [ -z "$DCAI_NUM_OFFLINE_EPS_TRAIN" ] && [ -n "$DCAI_NUM_OFFLINE_EPS_VAL" ]; then
        # DCAI_NUM_OFFLINE_EPS_TRAIN should be %90 of the whole data
        DCAI_NUM_OFFLINE_EPS_TRAIN=$((9 * DCAI_NUM_OFFLINE_EPS_VAL))
    fi

    DCAI_NUM_OFFLINE_EPS_TEST="${DCAI_NUM_OFFLINE_EPS_TEST:-100}"

    if [[ -n "$DCAI_START_LEVEL" && -n "$DCAI_NUM_LEVELS" ]]; then
        DCAI_START_SEED_TRAIN="$DCAI_START_LEVEL"
        DCAI_NUM_SEEDS_TRAIN="$DCAI_NUM_LEVELS"
        DCAI_START_SEED_VAL=$((DCAI_START_SEED_TRAIN + DCAI_NUM_SEEDS_TRAIN))
        DCAI_NUM_SEEDS_VAL="$DCAI_NUM_LEVELS"

    elif [[ -n "$DCAI_START_LEVEL" ]]; then
        DCAI_START_SEED_TRAIN="$DCAI_START_LEVEL"
        DCAI_NUM_SEEDS_TRAIN="$DCAI_NUM_LEVELS"
        DCAI_START_SEED_VAL=$((DCAI_START_SEED_TRAIN + DCAI_NUM_OFFLINE_EPS_TRAIN))
        DCAI_NUM_SEEDS_VAL="$DCAI_NUM_LEVELS"

    elif [[ -n "$DCAI_NUM_LEVELS" ]]; then
        DCAI_START_SEED_TRAIN=$((RANDOM))
        DCAI_NUM_SEEDS_TRAIN="$DCAI_NUM_LEVELS"
        DCAI_START_SEED_VAL=$((DCAI_START_SEED_TRAIN + DCAI_NUM_LEVELS))
        DCAI_NUM_SEEDS_VAL="$DCAI_NUM_LEVELS"

    else
        DCAI_START_SEED_TRAIN=-1
        DCAI_NUM_SEEDS_TRAIN=-1
        DCAI_START_SEED_VAL=-1
        DCAI_NUM_SEEDS_VAL=-1
    fi

    DCAI_START_SEED_TEST=$DCAI_START_SEED_TRAIN
    DCAI_NUM_SEEDS_TEST=$DCAI_NUM_SEEDS_TRAIN
}

# This function mostly should be used before all other handle_[*]() functions
handle_offline_data_vars() {
    # If $DCAI_DATA_DIR is not already assigned in stages other than 'qstar' and 'ucs', the value of $DCAI_ENV will be used
    if [ -z "$DCAI_DATA_DIR" ] && [ "$DCAI_STAGE" != "qstar" ] && [ "$DCAI_STAGE" != "ucs" ]; then
        DCAI_DATA_DIR="$DCAI_ENV"
        echo "WARNING: The argument '--data_dir' was not given. By default, the environment name ($DCAI_ENV) will be used as the value to this argument."
        echo ""
    fi

    DCAI_OFFLINE_DIR=deepcubeai/data/${DCAI_DATA_DIR}/offline
    DCAI_OFFLINE_ENC_DIR=deepcubeai/data/${DCAI_DATA_DIR}/offline_enc
    DCAI_OFFLINE_ENV_TEST_DIR=deepcubeai/data/${DCAI_DATA_DIR}/model_test
    DCAI_OFFLINE_SEARCH_TEST_DIR=deepcubeai/data/${DCAI_DATA_DIR}/search_test
    DCAI_DATA_SAMPLE_IMG_DIR=deepcubeai/data/${DCAI_DATA_DIR}/sample_images

    # Set DCAI_TRAIN_DATA_FILE_NAME to "train_data" if DCAI_DATA_FILE_NAME is empty,
    # else append "_train_data" unless it already contains "train_data"
    DCAI_TRAIN_DATA_FILE_NAME=$([ -z "$DCAI_DATA_FILE_NAME" ] && echo "train_data" || (echo "$DCAI_DATA_FILE_NAME" | grep -q "train_data" && echo "$DCAI_DATA_FILE_NAME" || echo "${DCAI_DATA_FILE_NAME}_train_data"))
    DCAI_OFFLINE_TRAIN="${DCAI_OFFLINE_DIR}/${DCAI_TRAIN_DATA_FILE_NAME}.pkl"

    # Set DCAI_VAL_DATA_FILE_NAME to "val_data" if DCAI_DATA_FILE_NAME is empty,
    # else append "_val_data" unless it already contains "val_data"
    DCAI_VAL_DATA_FILE_NAME=$([ -z "$DCAI_DATA_FILE_NAME" ] && echo "val_data" || (echo "$DCAI_DATA_FILE_NAME" | grep -q "val_data" && echo "$DCAI_DATA_FILE_NAME" || echo "${DCAI_DATA_FILE_NAME}_val_data"))
    DCAI_OFFLINE_VAL=${DCAI_OFFLINE_DIR}/${DCAI_VAL_DATA_FILE_NAME}.pkl

    # Set DCAI_ENV_TEST_DATA_FILE_NAME to "env_test_data" if DCAI_DATA_FILE_NAME is empty,
    # else append "_env_test_data" unless it already contains "env_test_data"
    DCAI_ENV_TEST_DATA_FILE_NAME=$([ -z "$DCAI_DATA_FILE_NAME" ] && echo "env_test_data" || (echo "$DCAI_DATA_FILE_NAME" | grep -q "env_test_data" && echo "$DCAI_DATA_FILE_NAME" || echo "${DCAI_DATA_FILE_NAME}_env_test_data"))
    DCAI_OFFLINE_ENV_TEST=$DCAI_OFFLINE_ENV_TEST_DIR/${DCAI_ENV_TEST_DATA_FILE_NAME}.pkl

    # Set DCAI_SEARCH_TEST_DATA_FILE_NAME to "search_test_data" if DCAI_DATA_FILE_NAME is empty,
    # else append "_search_test_data" unless it already contains "search_test_data"
    DCAI_SEARCH_TEST_DATA_FILE_NAME=$([ -z "$DCAI_DATA_FILE_NAME" ] && echo "search_test_data" || (echo "$DCAI_DATA_FILE_NAME" | grep -q "search_test_data" && echo "$DCAI_DATA_FILE_NAME" || echo "${DCAI_DATA_FILE_NAME}_search_test_data"))
    DCAI_OFFLINE_SEARCH_TEST=$DCAI_OFFLINE_SEARCH_TEST_DIR/${DCAI_SEARCH_TEST_DATA_FILE_NAME}.pkl

    # Set DCAI_TRAIN_ENC_DATA_FILE_NAME to "train_data_enc" if DCAI_DATA_FILE_NAME is empty,
    # else append "_train_data_enc" unless it already contains "train_data_enc"
    DCAI_TRAIN_ENC_DATA_FILE_NAME=$([ -z "$DCAI_DATA_FILE_NAME" ] && echo "train_data_enc" || (echo "$DCAI_DATA_FILE_NAME" | grep -q "train_data_enc" && echo "$DCAI_DATA_FILE_NAME" || echo "${DCAI_DATA_FILE_NAME}_train_data_enc"))
    DCAI_OFFLINE_TRAIN_ENC=${DCAI_OFFLINE_DIR}/${DCAI_TRAIN_ENC_DATA_FILE_NAME}.pkl

    # Set DCAI_VAL_ENC_DATA_FILE_NAME to "val_data_enc" if DCAI_DATA_FILE_NAME is empty,
    # else append "_val_data_enc" unless it already contains "val_data_enc"
    DCAI_VAL_ENC_DATA_FILE_NAME=$([ -z "$DCAI_DATA_FILE_NAME" ] && echo "val_data_enc" || (echo "$DCAI_DATA_FILE_NAME" | grep -q "val_data_enc" && echo "$DCAI_DATA_FILE_NAME" || echo "${DCAI_DATA_FILE_NAME}_val_data_enc"))
    DCAI_OFFLINE_VAL_ENC=${DCAI_OFFLINE_DIR}/${DCAI_VAL_ENC_DATA_FILE_NAME}.pkl

    # Set DCAI_VAL_ENC_DATA_FILE_NAME to "val_data_enc" if DCAI_DATA_FILE_NAME is empty,
    # else append "_val_data_enc" unless it already contains "val_data_enc"
    DCAI_VAL_ENC_DATA_FILE_NAME=$([ -z "$DCAI_DATA_FILE_NAME" ] && echo "val_data_enc" || (echo "$DCAI_DATA_FILE_NAME" | grep -q "val_data_enc" && echo "$DCAI_DATA_FILE_NAME" || echo "${DCAI_DATA_FILE_NAME}_val_data_enc"))
    DCAI_OFFLINE_VAL_ENC=${DCAI_OFFLINE_DIR}/${DCAI_VAL_ENC_DATA_FILE_NAME}.pkl

    DCAI_NUM_CPUS=${DCAI_NUM_CPUS:-1}
}

handle_search_test_data_vars() {
    DCAI_NUM_OFFLINE_STEPS=${DCAI_NUM_OFFLINE_STEPS:- -1}
}

handle_model_test_vars() {
    DCAI_OFFLINE_ENV_TEST="${DCAI_MODEL_TEST_DATA:-$DCAI_OFFLINE_ENV_TEST}"
    if [ -z "$DCAI_MODEL_TEST_DATA" ]; then
        echo ""
        echo "=========== WARNING ==========="
        echo "The argument '--model_test_data_dir' was not given. By default, the following path will be used:"
        echo "$DCAI_OFFLINE_ENV_TEST"
        echo "==============================="
        echo ""
    fi
    DCAI_PRINT_INTERVAL="${DCAI_PRINT_INTERVAL:-1}"
}

handle_disc_vs_cont_vars() {
    # Set DCAI_NUM_EPISODES to -1 if it's empty or unset
    # Setting DCAI_NUM_EPISODES to -1 means it will use all the episodes present in dataset
    DCAI_NUM_EPISODES="${DCAI_NUM_EPISODES:- -1}"
    # Set DCAI_NUM_STEPS to -1 if it's empty or unset
    # Setting DCAI_NUM_STEPS to -1 means it will use all the steps present in dataset
    DCAI_NUM_STEPS="${DCAI_NUM_STEPS:- -1}"
    DCAI_SAVE_DIR="${DCAI_SAVE_DIR:-$(pwd)}"
    DCAI_PRINT_INTERVAL="${DCAI_PRINT_INTERVAL:-1}"
}

handle_env_model_vars() {
    DCAI_ENV_MODEL_SAVE_DIR=deepcubeai/saved_env_models
    DCAI_ENV_MODEL_DIR=${DCAI_ENV_MODEL_SAVE_DIR}/${DCAI_ENV_MODEL_NAME}/
    # If $DCAI_ENV_TRAIN_BATCH_SIZE is not given as an arg, set it to the default value (100)
    DCAI_ENV_TRAIN_BATCH_SIZE=${DCAI_ENV_TRAIN_BATCH_SIZE:-100}
    # For continous model - Maximum number of steps to predict
    DCAI_NUM_ENV_TRAIN_STEPS=1
}

# Function to set up MPI environment and construct options
setup_mpi() {
    export OMPI_MCA_OPAL_CUDA_SUPPORT=true
    HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST)
    NUM_NODES=$SLURM_JOB_NUM_NODES
    NUM_WORKERS_PER_NODE=$(echo $SLURM_JOB_GPUS | awk -F, '{print NF}') #$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    TOTAL_WORKERS=0
    H_OPTION=""

    # Construct -H option for mpirun
    for HOST in $HOSTS; do
        # # Get the IP address of the current host
        # IP_ADDRESS=$(getent ahosts $HOST | head -n 1 | awk '{ print $1 }')
        if [ -z "$H_OPTION" ]; then
            NUM_WORKERS=$((NUM_WORKERS_PER_NODE)) # + 1))
        else
            NUM_WORKERS=$NUM_WORKERS_PER_NODE
        fi
        H_OPTION+="$HOST:$NUM_WORKERS,"
        TOTAL_WORKERS=$((TOTAL_WORKERS + NUM_WORKERS))
    done
    H_OPTION=${H_OPTION::-1} # Remove the last comma

    # Set MASTER_ADDR and MASTER_PORT
    MASTER_ADDR=$(echo "$H_OPTION" | cut -d, -f1 | cut -d: -f1)
    MASTER_PORT=$(shuf -i 2000-65000 -n 1) # Generate a random port number between 2000 and 65000

    NP_OPTION=$TOTAL_WORKERS #$((NUM_NODES * NUM_WORKERS_PER_NODE))
    # export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK - 2))
    # export I_MPI_PIN_DOMAIN=omp
    # export NCCL_DEBUG=INFO

    # Output the MPI run command parameters
    echo "Variables for running MPI job:"
    echo "Nodes: $HOSTS"
    echo "MASTER_ADDR=$MASTER_ADDR"
    echo "MASTER_PORT=$MASTER_PORT"
    echo "-H = $H_OPTION"
    echo "-np = $NP_OPTION"
    echo ""
}

handle_heur_model_vars() {
    DCAI_HEUR_MODEL_SAVE_DIR=deepcubeai/saved_heur_models
    DCAI_HEUR_MODEL_DIR=${DCAI_HEUR_MODEL_SAVE_DIR}/${DCAI_HEUR_NNET_NAME}/current
    # Use DDP (single node multi-gpu / multiple nodes multi-gpu)
    DCAI_USE_DIST=${DCAI_USE_DIST:-false}
    # Number of test states for testing DQN after training
    DCAI_NUM_TEST=${DCAI_NUM_TEST:-1000}
    [ "$DCAI_USE_DIST" = "true" ] && setup_mpi
}

handle_qstar_vars() {
    DCAI_OFFLINE_SEARCH_TEST="${DCAI_SEARCH_TEST_DATA:-$DCAI_OFFLINE_SEARCH_TEST}"
    if [ -z "$DCAI_SEARCH_TEST_DATA" ]; then
        echo "WARNING: The argument '--search_test_data' was not given. By default, the following path will be used:"
        echo "$DCAI_OFFLINE_SEARCH_TEST"
    fi
    DCAI_SAVE_IMGS="${DCAI_SAVE_IMGS:-False}"
    DCAI_QSTAR_H_WEIGHT="${DCAI_QSTAR_H_WEIGHT:-1}"
    # Set DCAI_RESULTS_DIR to DCAI_QSTAR_RESULTS_DIR if '--qstar_results_dir' is given as an arg, otherwise use the default path.
    DCAI_RESULTS_DIR="deepcubeai/results/${DCAI_ENV}/${DCAI_QSTAR_RESULTS_DIR:-"model=${DCAI_ENV_MODEL_NAME}_heur=${DCAI_HEUR_NNET_NAME}_QSTAR_results/path_cost_weight=${DCAI_QSTAR_WEIGHT}__h_weight=${DCAI_QSTAR_H_WEIGHT}"}"
}

handle_ucs_vars() {
    DCAI_OFFLINE_SEARCH_TEST="${DCAI_SEARCH_TEST_DATA:-$DCAI_OFFLINE_SEARCH_TEST}"
    if [ -z "$DCAI_SEARCH_TEST_DATA" ]; then
        echo "WARNING: The argument '--search_test_data' was not given. By default, the following path will be used:"
        echo "$DCAI_OFFLINE_SEARCH_TEST"
    fi
    DCAI_SAVE_IMGS="${DCAI_SAVE_IMGS:-False}"
    # Set DCAI_RESULTS_DIR to DCAI_UCS_RESULTS_DIR if '--ucs_results_dir' is given as an arg, otherwise use the default path.
    DCAI_RESULTS_DIR="deepcubeai/results/${DCAI_ENV}/${DCAI_UCS_RESULTS_DIR:-"model=${DCAI_ENV_MODEL_NAME}_UCS_results"}"
}

handle_gbfs_vars() {
    DCAI_OFFLINE_SEARCH_TEST="${DCAI_SEARCH_TEST_DATA:-$DCAI_OFFLINE_SEARCH_TEST}"
    if [ -z "$DCAI_SEARCH_TEST_DATA" ]; then
        echo "WARNING: The argument '--search_test_data' was not given. By default, the following path will be used:"
        echo "$DCAI_OFFLINE_SEARCH_TEST"
    fi
    # Set DCAI_RESULTS_DIR to DCAI_GBFS_RESULTS_DIR if '--gbfs_results_dir' is given as an arg, otherwise use the default path.
    DCAI_RESULTS_DIR="deepcubeai/results/${DCAI_ENV}/${DCAI_GBFS_RESULTS_DIR:-"model=${DCAI_ENV_MODEL_NAME}_heur=${DCAI_HEUR_NNET_NAME}_GBFS_results"}"
}

handle_data_viz_vars() {
    DCAI_NUM_TRAIN_TRAJS_VIZ="${DCAI_NUM_TRAIN_TRAJS_VIZ:-8}"
    DCAI_NUM_TRAIN_STEPS_VIZ="${DCAI_NUM_TRAIN_STEPS_VIZ:-2}"
    DCAI_NUM_VAL_TRAJS_VIZ="${DCAI_NUM_VAL_TRAJS_VIZ:-8}"
    DCAI_NUM_VAL_STEPS_VIZ="${DCAI_NUM_VAL_STEPS_VIZ:-2}"
}

check_variables() {
    for var in "$@"; do
        if [ -z "${!var}" ]; then
            echo "-----------"
            echo "ERROR: One or more required arguments are missing for the selected stage."
            echo "-----------"
            unset_vars
            exit 1
        fi
    done
}

if [ "$DCAI_STAGE" == "gen_offline" ]; then
    handle_offline_data_args
    handle_offline_data_vars
    check_variables "DCAI_STAGE" "DCAI_ENV" "DCAI_NUM_OFFLINE_EPS_TRAIN" "DCAI_NUM_OFFLINE_STEPS" "DCAI_OFFLINE_TRAIN" "DCAI_NUM_CPUS" "DCAI_START_SEED_TRAIN" "DCAI_NUM_SEEDS_TRAIN" "DCAI_NUM_OFFLINE_EPS_VAL" "DCAI_OFFLINE_VAL" "DCAI_START_SEED_VAL" "DCAI_NUM_SEEDS_VAL"

elif [ "$DCAI_STAGE" == "gen_env_test" ]; then
    handle_offline_data_args
    handle_offline_data_vars
    check_variables "DCAI_STAGE" "DCAI_ENV" "DCAI_NUM_OFFLINE_EPS_TEST" "DCAI_NUM_OFFLINE_STEPS" "DCAI_OFFLINE_ENV_TEST" "DCAI_NUM_CPUS" "DCAI_START_SEED_TEST" "DCAI_NUM_SEEDS_TEST"

elif [ "$DCAI_STAGE" == "gen_search_test" ]; then
    handle_offline_data_args
    handle_offline_data_vars
    handle_search_test_data_vars
    check_variables "DCAI_STAGE" "DCAI_ENV" "DCAI_NUM_OFFLINE_EPS_TEST" "DCAI_NUM_OFFLINE_STEPS" "DCAI_OFFLINE_SEARCH_TEST" "DCAI_START_SEED_TEST"

elif [ "$DCAI_STAGE" == "train_model" ]; then
    handle_offline_data_vars
    handle_env_model_vars
    check_variables "DCAI_STAGE" "DCAI_ENV" "DCAI_OFFLINE_TRAIN" "DCAI_OFFLINE_VAL" "DCAI_ENV_TRAIN_BATCH_SIZE" "DCAI_ENV_MODEL_NAME" "DCAI_ENV_MODEL_SAVE_DIR"

elif [ "$DCAI_STAGE" == "train_model_cont" ]; then
    handle_offline_data_vars
    handle_env_model_vars
    check_variables "DCAI_STAGE" "DCAI_ENV" "DCAI_OFFLINE_TRAIN" "DCAI_OFFLINE_VAL" "DCAI_ENV_TRAIN_BATCH_SIZE" "DCAI_ENV_MODEL_NAME" "DCAI_ENV_MODEL_SAVE_DIR" "DCAI_NUM_ENV_TRAIN_STEPS"

elif [ "$DCAI_STAGE" == "test_model" ]; then
    handle_offline_data_vars
    handle_model_test_vars
    handle_env_model_vars
    check_variables "DCAI_STAGE" "DCAI_ENV" "DCAI_OFFLINE_ENV_TEST" "DCAI_ENV_MODEL_DIR" "DCAI_PRINT_INTERVAL"

elif [ "$DCAI_STAGE" == "test_model_cont" ]; then
    handle_offline_data_vars
    handle_model_test_vars
    handle_env_model_vars
    check_variables "DCAI_STAGE" "DCAI_ENV" "DCAI_OFFLINE_ENV_TEST" "DCAI_ENV_MODEL_DIR" "DCAI_PRINT_INTERVAL"

elif [ "$DCAI_STAGE" == "encode_offline" ]; then
    handle_offline_data_vars
    handle_env_model_vars
    check_variables "DCAI_STAGE" "DCAI_ENV" "DCAI_OFFLINE_TRAIN" "DCAI_OFFLINE_TRAIN_ENC" "DCAI_OFFLINE_VAL" "DCAI_OFFLINE_VAL_ENC" "DCAI_ENV_MODEL_DIR"

elif [ "$DCAI_STAGE" == "train_heur" ]; then
    handle_offline_data_vars
    handle_heur_model_vars
    handle_env_model_vars
    check_variables "DCAI_STAGE" "DCAI_ENV" "DCAI_HEUR_NNET_NAME" "DCAI_HEUR_MODEL_SAVE_DIR" "DCAI_ENV_MODEL_DIR" "DCAI_OFFLINE_TRAIN_ENC" "DCAI_OFFLINE_VAL_ENC" "DCAI_PER_EQ_TOL" "DCAI_HEUR_BATCH_SIZE" "DCAI_STATES_PER_UPDATE" "DCAI_MAX_SOLVE_STEPS" "DCAI_START_STEPS" "DCAI_GOAL_STEPS" "DCAI_NUM_TEST" "DCAI_USE_DIST"

elif [ "$DCAI_STAGE" == "qstar" ]; then
    handle_offline_data_vars
    handle_heur_model_vars
    handle_env_model_vars
    handle_qstar_vars
    check_variables "DCAI_STAGE" "DCAI_ENV" "DCAI_OFFLINE_SEARCH_TEST" "DCAI_HEUR_MODEL_DIR" "DCAI_ENV_MODEL_DIR" "DCAI_QSTAR_BATCH_SIZE" "DCAI_QSTAR_WEIGHT" "DCAI_QSTAR_H_WEIGHT" "DCAI_RESULTS_DIR" "DCAI_PER_EQ_TOL" "DCAI_SAVE_IMGS"

elif [ "$DCAI_STAGE" == "ucs" ]; then
    handle_offline_data_vars
    handle_env_model_vars
    handle_ucs_vars
    check_variables "DCAI_STAGE" "DCAI_ENV" "DCAI_OFFLINE_SEARCH_TEST" "DCAI_ENV_MODEL_DIR" "DCAI_UCS_BATCH_SIZE" "DCAI_RESULTS_DIR" "DCAI_PER_EQ_TOL" "DCAI_SAVE_IMGS"

elif [ "$DCAI_STAGE" == "gbfs" ]; then
    handle_offline_data_vars
    handle_heur_model_vars
    handle_env_model_vars
    handle_gbfs_vars
    check_variables "DCAI_ENV" "DCAI_OFFLINE_SEARCH_TEST" "DCAI_HEUR_MODEL_DIR" "DCAI_ENV_MODEL_DIR" "DCAI_RESULTS_DIR" "DCAI_PER_EQ_TOL" "DCAI_SEARCH_ITRS"

elif [ "$DCAI_STAGE" == "visualize_data" ]; then
    handle_offline_data_vars
    handle_data_viz_vars
    check_variables "DCAI_STAGE" "DCAI_ENV" "DCAI_OFFLINE_TRAIN" "DCAI_OFFLINE_VAL" "DCAI_NUM_TRAIN_TRAJS_VIZ" "DCAI_NUM_TRAIN_STEPS_VIZ" "DCAI_NUM_VAL_TRAJS_VIZ" "DCAI_NUM_VAL_STEPS_VIZ" "DCAI_DATA_SAMPLE_IMG_DIR"

elif [ "$DCAI_STAGE" == "disc_vs_cont" ]; then
    handle_offline_data_vars
    handle_disc_vs_cont_vars
    check_variables "DCAI_STAGE" "DCAI_ENV" "DCAI_OFFLINE_ENV_TEST" "DCAI_ENV_MODEL_DIR_DISC" "DCAI_ENV_MODEL_DIR_CONT" "DCAI_NUM_EPISODES" "DCAI_NUM_STEPS" "DCAI_SAVE_DIR" "DCAI_PRINT_INTERVAL"

else
    echo "Invalid stage name: $DCAI_STAGE"
    unset_vars
    exit 1
fi
