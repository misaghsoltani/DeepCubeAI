from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from multiprocessing.context import SpawnContext, SpawnProcess
from pathlib import Path
import pickle
import sys
import time
from typing import Any, TypeAlias

import numpy as np
from numpy import float32, uint8
from numpy.typing import NDArray
from torch.multiprocessing import Queue, get_context

from deepcubeai.environments.environment_abstract import Environment, State
from deepcubeai.utils import env_utils
from deepcubeai.utils.data_utils import Logger, get_file_path_without_extension, print_args

StateTrajType: TypeAlias = list[State]
StateImgTrajType: TypeAlias = NDArray[uint8 | float32]
ActionTrajType: TypeAlias = list[int]
StatesQueueType: TypeAlias = tuple[StateTrajType, ActionTrajType] | None
ImgsQueueType: TypeAlias = tuple[StateImgTrajType, ActionTrajType]


@dataclass(frozen=True, slots=True)
class GenerateOfflineConfig:
    """Config for generate_offline_data entrypoint."""

    env: str
    num_episodes: int
    num_steps: int
    data_file: str
    num_procs: int = 1
    start_level: int = -1
    num_levels: int = -1

    @staticmethod
    def from_json(path: str | Path) -> GenerateOfflineConfig:
        """Load config from JSON file path."""
        raw: dict[str, Any] = json.loads(Path(path).read_bytes())
        return GenerateOfflineConfig(**raw)


def viz_runner(
    state_traj_queue: Queue[StatesQueueType], state_img_traj_queue: Queue[ImgsQueueType], env_name: str
) -> None:
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
            # end-of-stream sentinel
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
    parser.add_argument("--start_level", type=int, default=-1, help="The seed for the starting level")
    parser.add_argument("--num_levels", type=int, default=-1, help="Number of levels to get the data from")
    return parser


def initialize_environment(env_name: str) -> Environment:
    """Initializes the environment based on the given name.

    Args:
        env_name (str): Name of the environment.

    Returns:
        Environment: The initialized environment.
    """
    return env_utils.get_environment(env_name)


def generate_episodes(
    env: Environment, num_episodes: int, num_steps: int, start_level: int, num_levels: int
) -> tuple[list[StateTrajType], list[ActionTrajType]]:
    """Generates episodes for the given environment.

    Args:
        env (Environment): The environment instance.
        num_episodes (int): Number of episodes to generate.
        num_steps (int): Number of steps per episode.
        start_level (int): The seed for the starting level.
        num_levels (int): Number of levels to get the data from.

    Returns:
        tuple[list[StateTrajType], list[ActionTrajType]]: State trajectories and action trajectories.
    """
    print("Getting episodes")
    start_time = time.time()
    state_trajs: list[StateTrajType]
    action_trajs: list[ActionTrajType]
    _, _, state_trajs, action_trajs = env.generate_episodes([num_steps] * num_episodes, start_level, num_levels)
    print(f"Time: {time.time() - start_time}\n")
    return state_trajs, action_trajs


def start_image_processes(
    num_procs: int, env_name: str, state_traj_queue: Queue[StatesQueueType], state_img_traj_queue: Queue[ImgsQueueType]
) -> list[SpawnProcess]:
    """Starts image processing subprocesses.

    Args:
        num_procs (int): Number of processors.
        env_name (str): Name of the environment.
        state_traj_queue (Queue): Queue containing state trajectories.
        state_img_traj_queue (Queue): Queue to put state image trajectories.

    Returns:
        list[Process]: List of started processes.
    """
    ctx: SpawnContext = get_context("spawn")
    procs: list[SpawnProcess] = []
    for _ in range(num_procs):
        proc: SpawnProcess = ctx.Process(target=viz_runner, args=(state_traj_queue, state_img_traj_queue, env_name))
        proc.daemon = True
        proc.start()
        procs.append(proc)
    return procs


def put_data_to_queues(
    state_trajs: list[StateTrajType], action_trajs: list[ActionTrajType], state_traj_queue: Queue[StatesQueueType]
) -> None:
    """Puts state and action trajectories into the queue.

    Args:
        state_trajs (list[StateTrajType]): List of state trajectories.
        action_trajs (list[ActionTrajType]): List of action trajectories.
        state_traj_queue (Queue): Queue to put the trajectories.
    """
    print("Putting data to queues")
    start_time = time.time()
    for state_traj, action_traj in zip(state_trajs, action_trajs, strict=False):
        state_traj_queue.put((state_traj, action_traj))
    print(f"Time: {time.time() - start_time}\n")


def get_images(
    num_episodes: int, state_img_traj_queue: Queue[ImgsQueueType], state_trajs: list[StateTrajType]
) -> tuple[list[StateImgTrajType], list[ActionTrajType]]:
    """Gets images from the state image trajectory queue.

    Args:
        num_episodes (int): Number of episodes.
        state_img_traj_queue (Queue): Queue containing state image trajectories.
        state_trajs (list[StateTrajType]): List of state trajectories.

    Returns:
        tuple[list[NDArray], list[ActionTrajType]]: State image trajectories and action trajectories.
    """
    print("Getting images")
    start_time = time.time()

    display_steps: ActionTrajType = list(np.linspace(1, num_episodes, 10, dtype=int))

    state_img_trajs: list[StateImgTrajType] = []
    action_trajs: list[ActionTrajType] = []
    for traj_num in range(len(state_trajs)):
        state_img_traj, action_traj = state_img_traj_queue.get()
        state_img_trajs.append(state_img_traj)
        action_trajs.append(action_traj)
        if traj_num in display_steps:
            print(f"{100 * traj_num / num_episodes:.2f}% (Total time: {time.time() - start_time:.2f})")
    print("")
    return state_img_trajs, action_trajs


def stop_processes(num_procs: int, state_traj_queue: Queue[StatesQueueType], procs: list[SpawnProcess]) -> None:
    """Stops the image processing subprocesses.

    Args:
        num_procs (int): Number of processors.
        state_traj_queue (Queue): Queue containing state trajectories.
        procs (list[Process]): List of processes to stop.
    """
    for _ in range(num_procs):
        state_traj_queue.put(None)
    for proc in procs:
        proc.join()


def save_data(data_file: str, state_img_trajs: list[StateImgTrajType], action_trajs: list[ActionTrajType]) -> None:
    """Saves the state image trajectories and action trajectories to a file.

    Args:
        data_file (str): Path to the data file.
        state_img_trajs (list[NDArray]): List of state image trajectories.
        action_trajs (list[ActionTrajType]): List of action trajectories.
    """
    data_path = Path(data_file)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    tmp_path = data_path.with_suffix(data_path.suffix + ".tmp")
    with open(tmp_path, "wb") as file:
        pickle.dump((state_img_trajs, action_trajs), file, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(data_path)
    print(f"Write time: {time.time() - start_time}")


def run_generate_offline(cfg: GenerateOfflineConfig) -> None:
    """Entry point for programmatic use (no stdout redirection)."""
    env: Environment = initialize_environment(cfg.env)
    state_trajs, action_trajs = generate_episodes(env, cfg.num_episodes, cfg.num_steps, cfg.start_level, cfg.num_levels)

    ctx: SpawnContext = get_context("spawn")
    state_traj_queue: Queue[StatesQueueType] = ctx.Queue()
    state_img_traj_queue: Queue[ImgsQueueType] = ctx.Queue()
    procs: list[SpawnProcess] = start_image_processes(cfg.num_procs, cfg.env, state_traj_queue, state_img_traj_queue)

    put_data_to_queues(state_trajs, action_trajs, state_traj_queue)
    state_img_trajs, action_trajs = get_images(cfg.num_episodes, state_img_traj_queue, state_trajs)

    stop_processes(cfg.num_procs, state_traj_queue, procs)
    save_data(cfg.data_file, state_img_trajs, action_trajs)


def main() -> None:
    """CLI entry point."""
    parser = parse_arguments()
    args = parser.parse_args()
    output_save_path_without_extension = get_file_path_without_extension(args.data_file)
    output_save_path = f"{output_save_path_without_extension}_info.txt"
    sys.stdout = Logger(output_save_path, "a")
    print_args(args)

    cfg = GenerateOfflineConfig(
        env=args.env,
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        data_file=args.data_file,
        num_procs=args.num_procs,
        start_level=args.start_level,
        num_levels=args.num_levels,
    )
    run_generate_offline(cfg)
    print("Done")


if __name__ == "__main__":
    main()
