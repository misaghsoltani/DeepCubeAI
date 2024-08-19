import os
import pickle
import sys
import time
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
from torch.multiprocessing import Process, Queue, get_context

from deepcubeai.environments.environment_abstract import Environment, State
from deepcubeai.utils import env_utils
from deepcubeai.utils.data_utils import Logger, get_file_path_without_extension, print_args


def viz_runner(state_traj_queue: Queue, state_img_traj_queue: Queue, env_name: str) -> None:
    """Runs the visualization process for state trajectories.

    Args:
        state_traj_queue (Queue): Queue containing state trajectories.
        state_img_traj_queue (Queue): Queue to put state image trajectories.
        env_name (str): Name of the environment.
    """
    env: Environment = env_utils.get_environment(env_name)

    while True:
        data = state_traj_queue.get()
        if data is None:
            break

        state_traj, action_traj = data
        state_img_traj = env.state_to_real(state_traj)

        state_img_traj_queue.put((state_img_traj, action_traj))


def parse_arguments() -> ArgumentParser:
    """Parses command-line arguments.

    Returns:
        ArgumentParser: The argument parser with the defined arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment")
    parser.add_argument("--num_episodes", type=int, required=True, help="Number of episodes")
    parser.add_argument("--num_steps", type=int, required=True, help="Number of steps per episode")
    parser.add_argument("--data_file", type=str, required=True, help="Directory to save files")
    parser.add_argument("--num_procs", type=int, default=1, help="Number of processors")
    parser.add_argument("--start_level",
                        type=int,
                        default=-1,
                        help="The seed for the starting level")
    parser.add_argument("--num_levels",
                        type=int,
                        default=-1,
                        help="Number of levels to get the data from")
    return parser


def initialize_environment(env_name: str) -> Environment:
    """Initializes the environment based on the given name.

    Args:
        env_name (str): Name of the environment.

    Returns:
        Environment: The initialized environment.
    """
    return env_utils.get_environment(env_name)


def generate_episodes(env: Environment, num_episodes: int, num_steps: int, start_level: int,
                      num_levels: int) -> Tuple[List[List[State]], List[List[int]]]:
    """Generates episodes for the given environment.

    Args:
        env (Environment): The environment instance.
        num_episodes (int): Number of episodes to generate.
        num_steps (int): Number of steps per episode.
        start_level (int): The seed for the starting level.
        num_levels (int): Number of levels to get the data from.

    Returns:
        Tuple[List[List[State]], List[List[int]]]: State trajectories and action trajectories.
    """
    print("Getting episodes")
    start_time = time.time()
    state_trajs: List[List[State]]
    action_trajs: List[List[int]]
    _, _, state_trajs, action_trajs = env.generate_episodes([num_steps] * num_episodes,
                                                            start_level, num_levels)
    print(f"Time: {time.time() - start_time}\n")
    return state_trajs, action_trajs


def start_image_processes(num_procs: int, env_name: str, state_traj_queue: Queue,
                          state_img_traj_queue: Queue) -> List[Process]:
    """Starts image processing subprocesses.

    Args:
        num_procs (int): Number of processors.
        env_name (str): Name of the environment.
        state_traj_queue (Queue): Queue containing state trajectories.
        state_img_traj_queue (Queue): Queue to put state image trajectories.

    Returns:
        List[Process]: List of started processes.
    """
    ctx = get_context("spawn")
    procs: List[Process] = []
    for _ in range(num_procs):
        proc = ctx.Process(target=viz_runner,
                           args=(state_traj_queue, state_img_traj_queue, env_name))
        proc.daemon = True
        proc.start()
        procs.append(proc)
    return procs


def put_data_to_queues(state_trajs: List[List[State]], action_trajs: List[List[int]],
                       state_traj_queue: Queue) -> None:
    """Puts state and action trajectories into the queue.

    Args:
        state_trajs (List[List[State]]): List of state trajectories.
        action_trajs (List[List[int]]): List of action trajectories.
        state_traj_queue (Queue): Queue to put the trajectories.
    """
    print("Putting data to queues")
    start_time = time.time()
    for state_traj, action_traj in zip(state_trajs, action_trajs):
        state_traj_queue.put((state_traj, action_traj))
    print(f"Time: {time.time() - start_time}\n")


def get_images(num_episodes: int, state_img_traj_queue: Queue,
               state_trajs: List[List[State]]) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Gets images from the state image trajectory queue.

    Args:
        num_episodes (int): Number of episodes.
        state_img_traj_queue (Queue): Queue containing state image trajectories.
        state_trajs (List[List[State]]): List of state trajectories.

    Returns:
        Tuple[List[np.ndarray], List[List[int]]]: State image trajectories and action trajectories.
    """
    print("Getting images")
    start_time = time.time()

    display_steps: List[int] = list(np.linspace(1, num_episodes, 10, dtype=int))

    state_img_trajs: List[np.ndarray] = []
    action_trajs: List[List[int]] = []
    for traj_num in range(len(state_trajs)):
        state_img_traj, action_traj = state_img_traj_queue.get()
        state_img_trajs.append(state_img_traj)
        action_trajs.append(action_traj)
        if traj_num in display_steps:
            print(f"{100 * traj_num / num_episodes:.2f}% "
                  f"(Total time: {time.time() - start_time:.2f})")
    print("")
    return state_img_trajs, action_trajs


def stop_processes(num_procs: int, state_traj_queue: Queue, procs: List[Process]) -> None:
    """Stops the image processing subprocesses.

    Args:
        num_procs (int): Number of processors.
        state_traj_queue (Queue): Queue containing state trajectories.
        procs (List[Process]): List of processes to stop.
    """
    for _ in range(num_procs):
        state_traj_queue.put(None)
    for proc in procs:
        proc.join()


def save_data(data_file: str, state_img_trajs: List[np.ndarray],
              action_trajs: List[List[int]]) -> None:
    """Saves the state image trajectories and action trajectories to a file.

    Args:
        data_file (str): Path to the data file.
        state_img_trajs (List[np.ndarray]): List of state image trajectories.
        action_trajs (List[List[int]]): List of action trajectories.
    """
    data_dir = os.path.dirname(data_file)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    start_time = time.time()
    with open(data_file, "wb") as file:
        pickle.dump((state_img_trajs, action_trajs), file, protocol=-1)
    print(f"Write time: {time.time() - start_time}")


def main():
    """Main function to generate offline data."""
    parser = parse_arguments()
    args = parser.parse_args()
    output_save_path_without_extension = get_file_path_without_extension(args.data_file)
    output_save_path = f"{output_save_path_without_extension}_info.txt"
    sys.stdout = Logger(output_save_path, "a")
    print_args(args)

    env = initialize_environment(args.env)
    state_trajs, action_trajs = generate_episodes(env, args.num_episodes, args.num_steps,
                                                  args.start_level, args.num_levels)

    ctx = get_context("spawn")
    state_traj_queue = ctx.Queue()
    state_img_traj_queue = ctx.Queue()
    procs = start_image_processes(args.num_procs, args.env, state_traj_queue, state_img_traj_queue)

    put_data_to_queues(state_trajs, action_trajs, state_traj_queue)
    state_img_trajs, action_trajs = get_images(args.num_episodes, state_img_traj_queue,
                                               state_trajs)

    stop_processes(args.num_procs, state_traj_queue, procs)
    save_data(args.data_file, state_img_trajs, action_trajs)

    print("Done")


if __name__ == "__main__":
    main()
