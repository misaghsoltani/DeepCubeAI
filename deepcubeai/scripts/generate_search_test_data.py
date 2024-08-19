import os
import pickle
import sys
import time
from argparse import ArgumentParser
from inspect import signature
from typing import Dict, List

import numpy as np

from deepcubeai.environments.environment_abstract import Environment, State
from deepcubeai.utils import env_utils
from deepcubeai.utils.data_utils import Logger, get_file_path_without_extension, print_args


def parse_arguments() -> ArgumentParser:
    """Parses command-line arguments.

    Returns:
        ArgumentParser: The argument parser with the defined arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment")
    parser.add_argument("--num_episodes", type=int, required=True, help="Number of episodes")
    parser.add_argument("--num_steps", type=int, default=-1, help="Number of steps per episode")
    parser.add_argument("--data_file", type=str, required=True, help="Directory to save files")
    parser.add_argument("--start_level",
                        type=int,
                        default=-1,
                        help="The seed for the starting level")
    return parser


def generate_start_states(env: Environment, num_levels: int, start_level_seed: int) -> List[State]:
    """Generates start states for the environment.

    Args:
        env (Environment): The environment instance.
        num_levels (int): Number of levels to generate.
        start_level_seed (int): Seed for the starting level.

    Returns:
        List[State]: List of generated start states.
    """
    has_arg = "level_seeds" in signature(env.generate_start_states).parameters
    if has_arg:
        if start_level_seed < 0:
            start_level_seed = np.random.randint(0, 1000000)
        seeds_np = np.arange(start_level_seed, start_level_seed + num_levels)
        seeds_lst = seeds_np.tolist()
        return env.generate_start_states(num_levels, level_seeds=seeds_lst)

    return env.generate_start_states(num_levels)


def generate_goal_states(env: Environment, states: List[State], num_steps: int) -> List[State]:
    """Generates goal states for the environment.

    Args:
        env (Environment): The environment instance.
        states (List[State]): List of start states.
        num_steps (int): Number of steps to generate goal states.

    Returns:
        List[State]: List of generated goal states.
    """
    has_arg = "num_steps" in signature(env.get_goals).parameters
    if has_arg:
        num_steps = None if num_steps < 0 else num_steps
        return env.get_goals(states, num_steps)

    return env.get_goals(states)


def save_test_data(test_data_dict: Dict[str, List[State]], test_file_path: str) -> None:
    """Saves test data to a specified file.

    Args:
        test_data_dict (Dict[str, List[State]]): Dictionary containing test data.
        test_file_path (str): Path to save the test data.
    """
    test_file_dir = os.path.dirname(test_file_path)
    if not os.path.exists(test_file_dir):
        os.makedirs(test_file_dir)

    with open(test_file_path, "wb") as file:
        pickle.dump(test_data_dict, file)


def main():
    """Main function to generate and save search test data."""
    parser = parse_arguments()
    args = parser.parse_args()
    output_save_path_without_extension = get_file_path_without_extension(args.data_file)
    output_save_path = f"{output_save_path_without_extension}_info.txt"
    sys.stdout = Logger(output_save_path, "a")
    print_args(args)

    # initialize
    env: Environment = env_utils.get_environment(args.env)

    print("Generating the search test data.\n")
    print("Generating start states.")
    start_time = time.time()
    states = generate_start_states(env, args.num_episodes, args.start_level)
    print(f"Start states generated - Time {time.time() - start_time}")

    print("Generating goal states.")
    start_time = time.time()
    goal_states = generate_goal_states(env, states, args.num_steps)
    print(f"Goal states generated - Time {time.time() - start_time}")

    test_data_dict: Dict[str, List[State]] = {"states": states, "state_goals": goal_states}

    print("Writing test data into file.")
    start_time = time.time()
    save_test_data(test_data_dict, args.data_file)
    print(f"File saved successfully to: {args.data_file}")
    print(f"Write time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
