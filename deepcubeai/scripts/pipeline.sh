#!/bin/bash

# All the arguments and default values are handled in the 'pipeline_arg_handler.sh' file.
source deepcubeai/scripts/pipeline_arg_handler.sh

echo "Running stage $DCAI_STAGE"

if [ "$DCAI_STAGE" == "gen_offline" ]; then

       echo "Generating offline training data"
       python3 deepcubeai/scripts/generate_offline_data.py --env "$DCAI_ENV" --num_episodes "$DCAI_NUM_OFFLINE_EPS_TRAIN" --num_steps "$DCAI_NUM_OFFLINE_STEPS" --data_file "$DCAI_OFFLINE_TRAIN" --num_procs "$DCAI_NUM_CPUS" --start_level "$DCAI_START_SEED_TRAIN" --num_levels "$DCAI_NUM_SEEDS_TRAIN"
       echo ""
       echo "Generating offline validation data"
       python3 deepcubeai/scripts/generate_offline_data.py --env "$DCAI_ENV" --num_episodes "$DCAI_NUM_OFFLINE_EPS_VAL" --num_steps "$DCAI_NUM_OFFLINE_STEPS" --data_file "$DCAI_OFFLINE_VAL" --num_procs "$DCAI_NUM_CPUS" --start_level "$DCAI_START_SEED_VAL" --num_levels "$DCAI_NUM_SEEDS_VAL"

elif [ "$DCAI_STAGE" == "gen_env_test" ]; then

       echo "Generating environment model offline test data"
       python3 deepcubeai/scripts/generate_offline_data.py --env "$DCAI_ENV" --num_episodes "$DCAI_NUM_OFFLINE_EPS_TEST" --num_steps "$DCAI_NUM_OFFLINE_STEPS" --data_file "$DCAI_OFFLINE_ENV_TEST" --num_procs "$DCAI_NUM_CPUS" --start_level "$DCAI_START_SEED_TEST" --num_levels "$DCAI_NUM_SEEDS_TEST"

elif [ "$DCAI_STAGE" == "gen_search_test" ]; then

       echo "Generating search test data"
       python3 deepcubeai/scripts/generate_search_test_data.py --env "$DCAI_ENV" --num_episodes "$DCAI_NUM_OFFLINE_EPS_TEST" --num_steps "$DCAI_NUM_OFFLINE_STEPS" --data_file "$DCAI_OFFLINE_SEARCH_TEST" --start_level "$DCAI_START_SEED_TEST"

elif [ "$DCAI_STAGE" == "train_model" ]; then

       echo "Training model (discrete)"
       python3 deepcubeai/training/train_env_disc.py --env "$DCAI_ENV" --train_data "$DCAI_OFFLINE_TRAIN" --val_data "$DCAI_OFFLINE_VAL" --batch_size "$DCAI_ENV_TRAIN_BATCH_SIZE" --nnet_name "$DCAI_ENV_MODEL_NAME" --save_dir "$DCAI_ENV_MODEL_SAVE_DIR" --env_coeff 0.0001 --max_itrs 40000 --lr 0.001
       python3 deepcubeai/training/train_env_disc.py --env "$DCAI_ENV" --train_data "$DCAI_OFFLINE_TRAIN" --val_data "$DCAI_OFFLINE_VAL" --batch_size "$DCAI_ENV_TRAIN_BATCH_SIZE" --nnet_name "$DCAI_ENV_MODEL_NAME" --save_dir "$DCAI_ENV_MODEL_SAVE_DIR" --env_coeff 0.001 --max_itrs 60000 --lr 0.001
       python3 deepcubeai/training/train_env_disc.py --env "$DCAI_ENV" --train_data "$DCAI_OFFLINE_TRAIN" --val_data "$DCAI_OFFLINE_VAL" --batch_size "$DCAI_ENV_TRAIN_BATCH_SIZE" --nnet_name "$DCAI_ENV_MODEL_NAME" --save_dir "$DCAI_ENV_MODEL_SAVE_DIR" --env_coeff 0.01 --max_itrs 80000 --lr 0.001
       python3 deepcubeai/training/train_env_disc.py --env "$DCAI_ENV" --train_data "$DCAI_OFFLINE_TRAIN" --val_data "$DCAI_OFFLINE_VAL" --batch_size "$DCAI_ENV_TRAIN_BATCH_SIZE" --nnet_name "$DCAI_ENV_MODEL_NAME" --save_dir "$DCAI_ENV_MODEL_SAVE_DIR" --env_coeff 0.1 --max_itrs 100000 --lr 0.001
       python3 deepcubeai/training/train_env_disc.py --env "$DCAI_ENV" --train_data "$DCAI_OFFLINE_TRAIN" --val_data "$DCAI_OFFLINE_VAL" --batch_size "$DCAI_ENV_TRAIN_BATCH_SIZE" --nnet_name "$DCAI_ENV_MODEL_NAME" --save_dir "$DCAI_ENV_MODEL_SAVE_DIR" --env_coeff 0.5 --max_itrs 120000 --lr 0.001
       python3 deepcubeai/training/train_env_disc.py --env "$DCAI_ENV" --train_data "$DCAI_OFFLINE_TRAIN" --val_data "$DCAI_OFFLINE_VAL" --batch_size "$DCAI_ENV_TRAIN_BATCH_SIZE" --nnet_name "$DCAI_ENV_MODEL_NAME" --save_dir "$DCAI_ENV_MODEL_SAVE_DIR" --env_coeff 0.5 --max_itrs 140000 --lr 0.0001
       python3 deepcubeai/training/train_env_disc.py --env "$DCAI_ENV" --train_data "$DCAI_OFFLINE_TRAIN" --val_data "$DCAI_OFFLINE_VAL" --batch_size "$DCAI_ENV_TRAIN_BATCH_SIZE" --nnet_name "$DCAI_ENV_MODEL_NAME" --save_dir "$DCAI_ENV_MODEL_SAVE_DIR" --env_coeff 0.5 --max_itrs 160000 --lr 0.00001
       python3 deepcubeai/training/train_env_disc.py --env "$DCAI_ENV" --train_data "$DCAI_OFFLINE_TRAIN" --val_data "$DCAI_OFFLINE_VAL" --batch_size "$DCAI_ENV_TRAIN_BATCH_SIZE" --nnet_name "$DCAI_ENV_MODEL_NAME" --save_dir "$DCAI_ENV_MODEL_SAVE_DIR" --env_coeff 0.5 --max_itrs 180000 --lr 0.000001

elif [ "$DCAI_STAGE" == "train_model_cont" ]; then

       echo "Training model (continuous)"
       python3 deepcubeai/training/train_env_cont.py --env "$DCAI_ENV" --train_data "$DCAI_OFFLINE_TRAIN" --val_data "$DCAI_OFFLINE_VAL" --batch_size "$DCAI_ENV_TRAIN_BATCH_SIZE" --nnet_name "$DCAI_ENV_MODEL_NAME" --save_dir "$DCAI_ENV_MODEL_SAVE_DIR" --max_itrs 120000 --lr 0.001 --num_steps "$DCAI_NUM_ENV_TRAIN_STEPS"
       python3 deepcubeai/training/train_env_cont.py --env "$DCAI_ENV" --train_data "$DCAI_OFFLINE_TRAIN" --val_data "$DCAI_OFFLINE_VAL" --batch_size "$DCAI_ENV_TRAIN_BATCH_SIZE" --nnet_name "$DCAI_ENV_MODEL_NAME" --save_dir "$DCAI_ENV_MODEL_SAVE_DIR" --max_itrs 140000 --lr 0.0001 --num_steps "$DCAI_NUM_ENV_TRAIN_STEPS"
       python3 deepcubeai/training/train_env_cont.py --env "$DCAI_ENV" --train_data "$DCAI_OFFLINE_TRAIN" --val_data "$DCAI_OFFLINE_VAL" --batch_size "$DCAI_ENV_TRAIN_BATCH_SIZE" --nnet_name "$DCAI_ENV_MODEL_NAME" --save_dir "$DCAI_ENV_MODEL_SAVE_DIR" --max_itrs 160000 --lr 0.00001 --num_steps "$DCAI_NUM_ENV_TRAIN_STEPS"
       python3 deepcubeai/training/train_env_cont.py --env "$DCAI_ENV" --train_data "$DCAI_OFFLINE_TRAIN" --val_data "$DCAI_OFFLINE_VAL" --batch_size "$DCAI_ENV_TRAIN_BATCH_SIZE" --nnet_name "$DCAI_ENV_MODEL_NAME" --save_dir "$DCAI_ENV_MODEL_SAVE_DIR" --max_itrs 180000 --lr 0.000001 --num_steps "$DCAI_NUM_ENV_TRAIN_STEPS"

elif [ "$DCAI_STAGE" == "test_model" ]; then

       echo "Testing model (discrete)"
       python3 deepcubeai/scripts/test_model_disc.py --env "$DCAI_ENV" --data "$DCAI_OFFLINE_ENV_TEST" --env_dir "$DCAI_ENV_MODEL_DIR" --print_interval "$DCAI_PRINT_INTERVAL"

elif [ "$DCAI_STAGE" == "test_model_cont" ]; then

       echo "Testing model (continuous)"
       python3 deepcubeai/scripts/test_model_cont.py --env "$DCAI_ENV" --data "$DCAI_OFFLINE_ENV_TEST" --env_dir "$DCAI_ENV_MODEL_DIR" --print_interval "$DCAI_PRINT_INTERVAL"

elif [ "$DCAI_STAGE" == "encode_offline" ]; then

       echo "Encoding offline training data"
       python3 deepcubeai/scripts/encode_offline_data.py --env "$DCAI_ENV" --env_dir "$DCAI_ENV_MODEL_DIR" --data "$DCAI_OFFLINE_TRAIN" --data_enc "$DCAI_OFFLINE_TRAIN_ENC"
       echo ""
       echo "Encoding offline validation data"
       python3 deepcubeai/scripts/encode_offline_data.py --env "$DCAI_ENV" --env_dir "$DCAI_ENV_MODEL_DIR" --data "$DCAI_OFFLINE_VAL" --data_enc "$DCAI_OFFLINE_VAL_ENC"

elif [ "$DCAI_STAGE" == "train_heur" ]; then

       echo "Training heuristic function"
       if [ "$DCAI_USE_DIST" = "true" ]; then
              mpirun -np $NP_OPTION \
                     -H $H_OPTION \
                     -x MASTER_ADDR=$MASTER_ADDR \
                     -x MASTER_PORT=$MASTER_PORT \
                     -x PATH \
                     -bind-to none -map-by slot \
                     -mca pml ob1 -mca btl ^openib -mca orte_base_help_aggregate 0 \
                     python3 deepcubeai/training/qlearning_dist.py --env "$DCAI_ENV" --nnet_name "$DCAI_HEUR_NNET_NAME" --save_dir "$DCAI_HEUR_MODEL_SAVE_DIR" --env_model "$DCAI_ENV_MODEL_DIR" --train "$DCAI_OFFLINE_TRAIN_ENC" --val "$DCAI_OFFLINE_VAL_ENC" --per_eq_tol "$DCAI_PER_EQ_TOL" --batch_size "$DCAI_HEUR_BATCH_SIZE" --states_per_update "$DCAI_STATES_PER_UPDATE" --max_solve_steps "$DCAI_MAX_SOLVE_STEPS" --start_steps "$DCAI_START_STEPS" --goal_steps "$DCAI_GOAL_STEPS" --num_test "$DCAI_NUM_TEST"
       else
              python3 deepcubeai/training/qlearning.py --env "$DCAI_ENV" --nnet_name "$DCAI_HEUR_NNET_NAME" --save_dir "$DCAI_HEUR_MODEL_SAVE_DIR" --env_model "$DCAI_ENV_MODEL_DIR" --train "$DCAI_OFFLINE_TRAIN_ENC" --val "$DCAI_OFFLINE_VAL_ENC" --per_eq_tol "$DCAI_PER_EQ_TOL" --batch_size "$DCAI_HEUR_BATCH_SIZE" --states_per_update "$DCAI_STATES_PER_UPDATE" --max_solve_steps "$DCAI_MAX_SOLVE_STEPS" --start_steps "$DCAI_START_STEPS" --goal_steps "$DCAI_GOAL_STEPS" --num_test "$DCAI_NUM_TEST"
       fi

elif [ "$DCAI_STAGE" == "qstar" ]; then

       echo "Doing Q* search"
       python3 deepcubeai/search_methods/qstar_imag.py --env "$DCAI_ENV" --states "$DCAI_OFFLINE_SEARCH_TEST" --heur "$DCAI_HEUR_MODEL_DIR" --env_model "$DCAI_ENV_MODEL_DIR" --batch_size "$DCAI_QSTAR_BATCH_SIZE" --weight "$DCAI_QSTAR_WEIGHT" --results_dir "$DCAI_RESULTS_DIR" --per_eq_tol "$DCAI_PER_EQ_TOL" --save_imgs "$DCAI_SAVE_IMGS" --h_weight "$DCAI_QSTAR_H_WEIGHT"

elif [ "$DCAI_STAGE" == "ucs" ]; then

       echo "Doing uniform-cost search"
       python3 deepcubeai/search_methods/ucs_imag.py --env "$DCAI_ENV" --states "$DCAI_OFFLINE_SEARCH_TEST" --env_model "$DCAI_ENV_MODEL_DIR" --batch_size "$DCAI_UCS_BATCH_SIZE" --results_dir "$DCAI_RESULTS_DIR" --per_eq_tol "$DCAI_PER_EQ_TOL" --save_imgs "$DCAI_SAVE_IMGS"

elif [ "$DCAI_STAGE" == "gbfs" ]; then

       echo "Doing Greedy Best First Search"
       python3 deepcubeai/search_methods/gbfs_imag.py --env "$DCAI_ENV" --states "$DCAI_OFFLINE_SEARCH_TEST" --heur "$DCAI_HEUR_MODEL_DIR" --env_model "$DCAI_ENV_MODEL_DIR" --results_dir "$DCAI_RESULTS_DIR" --per_eq_tol "$DCAI_PER_EQ_TOL" --search_itrs "$DCAI_SEARCH_ITRS"

elif [ "$DCAI_STAGE" == "disc_vs_cont" ]; then

       echo "MSE Comparison (Discrete vs Continous)"
       python3 deepcubeai/extra/plot_disc_vs_cont.py --env "$DCAI_ENV" --model_test_data "$DCAI_OFFLINE_ENV_TEST" --env_model_dir_disc "$DCAI_ENV_MODEL_DIR_DISC" --env_model_dir_cont "$DCAI_ENV_MODEL_DIR_CONT" --num_episodes "$DCAI_NUM_EPISODES" --num_steps "$DCAI_NUM_STEPS" --save_dir "$DCAI_SAVE_DIR" --print_interval "$DCAI_PRINT_INTERVAL"

elif [ "$DCAI_STAGE" == "visualize_data" ]; then

       echo "Saving offline data as images"
       python3 deepcubeai/extra/offline_data_viz.py --env "$DCAI_ENV" --train_data "$DCAI_OFFLINE_TRAIN" --val_data "$DCAI_OFFLINE_VAL" --num_train_trajs "$DCAI_NUM_TRAIN_TRAJS_VIZ" --num_train_steps "$DCAI_NUM_TRAIN_STEPS_VIZ" --num_val_trajs "$DCAI_NUM_VAL_TRAJS_VIZ" --num_val_steps "$DCAI_NUM_VAL_STEPS_VIZ" --save_imgs "$DCAI_DATA_SAMPLE_IMG_DIR"

fi

unset_vars
