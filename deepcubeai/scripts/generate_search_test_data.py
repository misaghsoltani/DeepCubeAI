from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from inspect import signature
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, cast

import numpy as np

from deepcubeai.environments.cube3 import Cube3
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
    parser.add_argument("--start_level", type=int, default=-1, help="The seed for the starting level")
    # cube3-only flag: reversed search pairs (start=canonical, goal=scrambled)
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="cube3-only: generate reverse search data (start=canonical goal=scrambled)",
    )
    return parser


@dataclass(frozen=True, slots=True)
class GenerateSearchTestConfig:
    """Config for generate_search_test_data entrypoint."""

    env: str
    num_episodes: int
    data_file: str
    num_steps: int = -1
    start_level: int = -1
    reverse: bool = False

    @staticmethod
    def from_json(path: str | Path) -> GenerateSearchTestConfig:
        """Load config from a JSON file path."""
        raw: dict[str, Any] = json.loads(Path(path).read_bytes())
        return GenerateSearchTestConfig(**raw)


def generate_start_states(env: Environment, num_levels: int, start_level_seed: int) -> list[State]:
    """Generates start states for the environment.

    Args:
        env (Environment): The environment instance.
        num_levels (int): Number of levels to generate.
        start_level_seed (int): Seed for the starting level.

    Returns:
        list[State]: List of generated start states.
    """
    has_arg = "level_seeds" in signature(env.generate_start_states).parameters
    if has_arg:
        if start_level_seed < 0:
            start_level_seed = np.random.randint(0, 1000000)
        seeds_np = np.arange(start_level_seed, start_level_seed + num_levels)
        seeds_lst = seeds_np.tolist()
        return env.generate_start_states(num_levels, level_seeds=seeds_lst)

    return env.generate_start_states(num_levels)


def generate_goal_states(env: Environment, states: list[State], num_steps: int) -> list[State]:
    """Generates goal states for the environment.

    Args:
        env (Environment): The environment instance.
        states (list[State]): List of start states.
        num_steps (int): Number of steps to generate goal states.

    Returns:
        list[State]: List of generated goal states.
    """
    # Some environments define get_goals(states) and others get_goals(states, num_steps).
    # Avoid static inspection of optional abstract staticmethod. Try calling with the keyword,
    # else fall back to the single-arg form.
    get_goals = getattr(env, "get_goals", None)
    if get_goals is None:
        raise AttributeError("Environment does not implement 'get_goals'")

    num_steps_kw: int | None = None if num_steps < 0 else num_steps
    try:
        return get_goals(states, num_steps_kw)
    except TypeError:
        return get_goals(states)


def save_test_data(test_data_dict: dict[str, list[State]], test_file_path: str) -> None:
    """Saves test data to a specified file.

    Args:
        test_data_dict (dict[str, list[State]]): Dictionary containing test data.
        test_file_path (str): Path to save the test data.
    """
    p = Path(test_file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as file:
        pickle.dump(test_data_dict, file)
    tmp.replace(p)


def run_generate_search_test(cfg: GenerateSearchTestConfig) -> None:
    """Entry point for programmatic use (no stdout redirection)."""
    env: Environment = env_utils.get_environment(cfg.env)

    states: list[State] = generate_start_states(env, cfg.num_episodes, cfg.start_level)
    scrambled_states: list[State]
    goal_states: list[State]
    canonical_starts: list[State]
    if cfg.reverse:
        if env.env_name.lower() != "cube3":
            raise ValueError("--reverse is only supported for the 'cube3' environment")

        env_cube3: Cube3 = cast(Cube3, env)
        scrambled_states = states
        canonical_starts = cast(
            list[State], env_cube3._generate_canonical_goal_states(len(scrambled_states), np_format=False)
        )
        states = list(canonical_starts)
        goal_states = scrambled_states

    else:
        goal_states = generate_goal_states(env, states, cfg.num_steps)

    test_data_dict: dict[str, list[State]] = {"states": states, "state_goals": goal_states}
    save_test_data(test_data_dict, cfg.data_file)


def main() -> None:
    """Main function to generate and save search test data."""
    parser = parse_arguments()
    args = parser.parse_args()
    output_save_path_without_extension = get_file_path_without_extension(args.data_file)
    output_save_path = f"{output_save_path_without_extension}_info.txt"
    sys.stdout = Logger(output_save_path, "a")
    print_args(args)

    print("Generating the search test data.\n")
    cfg = GenerateSearchTestConfig(
        env=args.env,
        num_episodes=args.num_episodes,
        data_file=args.data_file,
        num_steps=args.num_steps,
        start_level=args.start_level,
        reverse=getattr(args, "reverse", False),
    )
    start_time = time.time()
    run_generate_search_test(cfg)
    print(f"File saved successfully to: {args.data_file}")
    print(f"Write time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
