show_help() {
    cat << EOF

===============================================================
DeepCubeAI - Learning Discrete World Models for Heuristic Search
Command-line interface for different stage of the DeepCubeAI.
For more details and examples, check out the GitHub repository:
https://github.com/misaghsoltani/DeepCubeAI
===============================================================


Usage: deepcubeai --stage <stage> --env <environment> [options]

Stages:
  1. gen_offline          Generate offline training data
  2. gen_env_test         Generate world model test data  
  3. gen_search_test      Generate search test data
  4. train_model          Train discrete world model
  5. test_model           Test discrete world model
  6. train_model_cont     Train continuous world model  
  7. test_model_cont      Test continuous world model
  8. disc_vs_cont         Compare discrete vs continuous models
  9. encode_offline       Encode offline data
  10. train_heur          Train heuristic network
  11. qstar               Run Q* search
  12. ucs                 Run Uniform Cost Search
  13. gbfs                Run Greedy Best-First Search
  14. visualize_data      Visualize training/validation data

Global Options:
  --env <environment>     Specify environment (e.g. cube3, sokoban, iceslider, digitjump)
  --data_dir <dir>        Data directory (default: <environment>)
  
Common Options:
  --data_file_name <name> Data file name prefix
  --env_model_name <name> Environment model name
  --heur_nnet_name <name> Heuristic network name
  --per_eq_tol <float>    Percent of latent state elements that need to be equal to declare equal. (default: 100)
  --num_cpus <int>        Number of CPUs to use

1. gen_offline
Generate offline training and validation data

Usage: deepcubeai --stage gen_offline --env <environment> [options]

Options:
  --num_offline_steps <int>   Number of steps for offline data
  --num_train_eps <int>       Number of training episodes  
  --num_val_eps <int>         Number of validation episodes
  --start_level <int>         Starting level for data generation
  --num_levels <int>          Number of levels to generate data for

Data directory structure:
deepcubeai
└── data
    └── <env_data_dir>
        └── offline
            ├── <data_file_name>_train_data.pkl
            └── <data_file_name>_val_data.pkl

2. gen_env_test  
Generate world model test data

Usage: deepcubeai --stage gen_env_test --env <environment> [options]

Options:
  --num_offline_steps <int>   Number of steps for test data
  --num_test_eps <int>        Number of test episodes (default: 1000)
  --start_level <int>         Starting level for test data
  --num_levels <int>          Number of levels for test data

Data directory structure:
deepcubeai
└── data
    └── <env_data_dir>
        └── model_test
            └── <test_file_name>_env_test_data.pkl

3. gen_search_test
Generate search test data 

Usage: deepcubeai --stage gen_search_test --env <environment> [options]

Options:
  --num_test_eps <int>        Number of search test episodes

Data directory structure:
deepcubeai
└── data
    └── <env_data_dir>
        └── search_test
            └── <search_test_file_name>_search_test_data.pkl

4. train_model
Train discrete world model

Usage: deepcubeai --stage train_model --env <environment> [options]

Options:
  --env_batch_size <int>      Batch size for training (default: 100)

Saved model directory structure:
deepcubeai
└── saved_env_models
    └── <disc_env_model_folder_name>
        ├── args.pkl
        ├── decoder_state_dict.pt
        ├── encoder_state_dict.pt
        ├── env_state_dict.pt
        ├── output.txt
        ├── train_itr.pkl
        └── pics
            ├── recon_itr0.jpg
            ├── recon_itr200.jpg
            └── ...

5. test_model 
Test discrete world model

Usage: deepcubeai --stage test_model --env <environment> [options]

Options:
  --model_test_data_dir <dir> Test data directory
  --print_interval <int>      Print interval (default: 1)

6. train_model_cont
Train continuous world model

Usage: deepcubeai --stage train_model_cont --env <environment> [options]

Options:
  --env_batch_size <int>      Batch size for training (default: 100)

Saved model directory structure:
deepcubeai
└── saved_env_models
    └── <cont_env_model_folder_name>
        ├── args.pkl
        ├── model_state_dict.pt
        ├── output.txt
        └── train_itr.pkl

7. test_model_cont
Test continuous world model

Usage: deepcubeai --stage test_model_cont --env <environment> [options]
  
Options:
  --model_test_data_dir <dir> Test data directory  
  --print_interval <int>      Print interval (default: 1)

8. disc_vs_cont
Compare discrete vs continuous models

Usage: deepcubeai --stage disc_vs_cont --env <environment> [options]

Options:
  --env_model_dir_disc <dir>  Discrete model directory
  --env_model_dir_cont <dir>  Continuous model directory  
  --save_dir <dir>            Directory to save comparison plot
  --num_steps <int>           Number of steps to compare (-1 for all)
  --num_episodes <int>        Number of episodes to compare (-1 for all)
  --print_interval <int>      Print interval (default: 1)

9. encode_offline
Encode offline data using trained model

Usage: deepcubeai --stage encode_offline --env <environment> [options]

Encoded data directory structure:
deepcubeai
└── data
    └── <env_data_dir>
        └── offline_enc
            ├── <data_file_name>_train_data_enc.pkl
            └── <data_file_name>_val_data_enc.pkl

10. train_heur
Train heuristic neural network

Usage: deepcubeai --stage train_heur --env <environment> [options]

Options:
  --heur_batch_size <int>     Batch size for heuristic training (default: 10000)
  --states_per_update <int>   States generated per update (default: 50000000)
  --start_steps <int>         Max steps from offline states for start states
  --goal_steps <int>          Max steps from start states for goal states
  --max_solve_steps <int>     Max steps for GBFS during training
  --use_dist                  Use distributed training

Saved heuristic model directory structure:
deepcubeai
└── saved_heur_models
    └── <heur_nnet_folder_name>
        ├── args.pkl
        ├── current
        │   ├── model_state_dict.pt
        │   └── status.pkl
        ├── output.txt
        └── target
            ├── model_state_dict.pt
            └── status.pkl

11. qstar
Run Q* search algorithm

Usage: deepcubeai --stage qstar --env <environment> [options]

Options:
  --qstar_batch_size <int>    Batch size for Q* search (default: 1)
  --qstar_weight <float>      Weight for path costs (default: 1)
  --qstar_results_dir <dir>   Directory to save results
  --save_imgs <bool>          Save solution path images (default: false)
  --search_test_data <path>   Custom search test data path

Results directory structure:
deepcubeai
└── results
    └── <environment>
        └── <results_dir>
            ├── output.txt
            ├── results.pkl
            └── qstar_soln_images
                ├── state_0.png
                ├── state_1.png
                └── ...

12. ucs
Run Uniform Cost Search

Usage: deepcubeai --stage ucs --env <environment> [options]

Options:
  --ucs_batch_size <int>      Batch size for UCS (default: 1)
  --ucs_results_dir <dir>     Directory to save results
  --save_imgs <bool>          Save solution path images (default: false)
  --search_test_data <path>   Custom search test data path

Results directory structure:
deepcubeai
└── results
    └── <environment>
        └── <results_dir>
            ├── output.txt
            ├── results.pkl
            └── ucs_soln_images
                ├── state_0.png
                ├── state_1.png
                └── ...

13. gbfs
Run Greedy Best-First Search

Usage: deepcubeai --stage gbfs --env <environment> [options]

Options:
  --gbfs_results_dir <dir>    Directory to save results
  --search_itrs <int>         Number of search iterations (default: 100)
  --search_test_data <path>   Custom search test data path

14. visualize_data
Visualize training and validation data

Usage: deepcubeai --stage visualize_data --env <environment> [options]

Options:
  --num_train_trajs_viz <int> Number of training trajectories (default: 8)
  --num_train_steps_viz <int> Steps per training trajectory (default: 2)
  --num_val_trajs_viz <int>   Number of validation trajectories (default: 8)
  --num_val_steps_viz <int>   Steps per validation trajectory (default: 2)

Note: For the cube3 and sokoban environments, levels are generated randomly, and the --start_level and --num_levels arguments are not used.

For more detailed information, please refer to the README.md file or visit the GitHub repository.
EOF
}