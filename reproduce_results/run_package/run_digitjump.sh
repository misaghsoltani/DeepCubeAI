ENV=digitjump
DATA_DIR=digitjump
ENV_MODEL_NAME_DISC=digitjump_disc
ENV_MODEL_NAME_CONT=digitjump_cont
ENV_MODEL_DIR_DISC=deepcubeai/saved_env_models/${ENV_MODEL_NAME_DISC}
ENV_MODEL_DIR_CONT=deepcubeai/saved_env_models/${ENV_MODEL_NAME_CONT}
HEUR_NNET_NAME=digitjump_heur
DATA_FILE_NAME_TRAIN_VAL=s0-1k_stp20
DATA_FILE_NAME_MODEL_TEST=s5k-5.1k_stp1k
DATA_FILE_NAME_MODEL_TEST_PLOT=s5k-5.1k_stp10k
DATA_FILE_NAME_SEARCH_TEST=s2k-2.1k
QSTAR_WEIGHT=0.7
QSTAR_H_WEIGHT=1.0
QSTAR_BATCH_SIZE=1
UCS_BATCH_SIZE=50
current_time=$(date +"%Y%m%d_%H%M%S")$(($(date +%N | bc)/1000000))
RESULTS_DIR_QSTAR="model=${ENV_MODEL_NAME_DISC}__heur=${HEUR_NNET_NAME}_QSTAR_results/path_cost_weight=${QSTAR_WEIGHT}__h_weight=${QSTAR_H_WEIGHT}__batchsize=${QSTAR_BATCH_SIZE}_${current_time}"
RESULTS_DIR_UCS="model=${ENV_MODEL_NAME_DISC}_UCS_results/batchsize=${UCS_BATCH_SIZE}_${current_time}"
RESULTS_DIR_GBFS="model=${ENV_MODEL_NAME_DISC}__heur=${HEUR_NNET_NAME}_GBFS_results/${current_time}"
PER_EQ_TOL=100
PLOTS_SAVE_DIR=deepcubeai
# Get the number of CPU cores available on the system
NUM_CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || wmic cpu get NumberOfCores 2>/dev/null)

deepcubeai --stage gen_offline --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_TRAIN_VAL --num_offline_steps 20 --num_train_eps 20000 --num_val_eps 5000 --num_cpus $NUM_CORES --start_level 0 --num_levels 1000
deepcubeai --stage gen_env_test --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_MODEL_TEST --num_offline_steps 1000 --num_test_eps 100 --num_cpus $NUM_CORES --start_level 5000 --num_levels 100
deepcubeai --stage gen_search_test --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_SEARCH_TEST --num_test_eps 100 --num_cpus $NUM_CORES --start_level 2000
deepcubeai --stage train_model --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_TRAIN_VAL --env_batch_size 100 --env_model_name $ENV_MODEL_NAME_DISC
deepcubeai --stage test_model --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_MODEL_TEST --env_model_name $ENV_MODEL_NAME_DISC --print_interval 50
deepcubeai --stage train_model_cont --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_TRAIN_VAL --env_batch_size 100 --env_model_name $ENV_MODEL_NAME_CONT
deepcubeai --stage test_model_cont --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_MODEL_TEST --env_model_name $ENV_MODEL_NAME_CONT --print_interval 50
deepcubeai --stage encode_offline --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_TRAIN_VAL --env_model_name $ENV_MODEL_NAME_DISC
deepcubeai --stage train_heur --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_TRAIN_VAL --env_model_name $ENV_MODEL_NAME_DISC --heur_nnet_name $HEUR_NNET_NAME --per_eq_tol $PER_EQ_TOL --heur_batch_size 10_000 --states_per_update 50_000_000 --start_steps 20 --goal_steps 20 --max_solve_steps 20 --num_test 1000
deepcubeai --stage qstar --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_SEARCH_TEST --env_model_name $ENV_MODEL_NAME_DISC --heur_nnet_name $HEUR_NNET_NAME --qstar_batch_size $QSTAR_BATCH_SIZE --qstar_weight $QSTAR_WEIGHT --qstar_h_weight $QSTAR_H_WEIGHT --per_eq_tol $PER_EQ_TOL --qstar_results_dir $RESULTS_DIR_QSTAR --save_imgs false
deepcubeai --stage ucs --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_SEARCH_TEST --env_model_name $ENV_MODEL_NAME_DISC --ucs_batch_size $UCS_BATCH_SIZE --per_eq_tol $PER_EQ_TOL --ucs_results_dir $RESULTS_DIR_UCS --save_imgs false
deepcubeai --stage gbfs --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_SEARCH_TEST --env_model_name $ENV_MODEL_NAME_DISC --heur_nnet_name $HEUR_NNET_NAME --per_eq_tol $PER_EQ_TOL --gbfs_results_dir $RESULTS_DIR_GBFS --search_itrs 100
deepcubeai --stage visualize_data --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_TRAIN_VAL --num_train_trajs_viz 8 --num_train_steps_viz 2 --num_val_trajs_viz 8 --num_val_steps_viz 2
deepcubeai --stage gen_env_test --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_MODEL_TEST_PLOT --num_offline_steps 10_000 --num_test_eps 100 --num_cpus $NUM_CORES --start_level 5000 --num_levels 100
deepcubeai --stage disc_vs_cont --env $ENV --data_dir $DATA_DIR --data_file_name $DATA_FILE_NAME_MODEL_TEST_PLOT --env_model_dir_disc $ENV_MODEL_DIR_DISC --env_model_dir_cont $ENV_MODEL_DIR_CONT --save_dir $PLOTS_SAVE_DIR --num_steps 1000 --num_episodes 100 --print_interval 1
