import os
import pickle
import sys
import time
from argparse import ArgumentParser, Namespace
from types import FunctionType
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import torch
from torch import nn

from deepcubeai.environments.environment_abstract import Environment, State
from deepcubeai.environments.sokoban import SokobanState
from deepcubeai.utils import data_utils, env_utils, misc_utils, nnet_utils
from deepcubeai.utils.imag_utils import random_walk


def gbfs(
    dqn: Union[nn.Module, FunctionType],
    env_model: Union[nn.Module, FunctionType],
    states_curr_np_inp: np.ndarray,
    states_goal_np_inp: np.ndarray,
    per_eq_tol: float,
    max_solve_steps: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy Best-First Search (GBFS) algorithm to solve states.

    Args:
        dqn (Union[nn.Module, FunctionType]): The DQN model or function.
        env_model (Union[nn.Module, FunctionType]): The environment model or function.
        states_curr_np_inp (np.ndarray): Current states as numpy array.
        states_goal_np_inp (np.ndarray): Goal states as numpy array.
        per_eq_tol (float): Percentage of elements that need to be equal to declare states equal.
        max_solve_steps (int): Maximum number of steps to solve.
        device (torch.device): Device to run the computations on.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Boolean array indicating if states are solved and array of
            number of steps taken.
    """
    states_curr_np = states_curr_np_inp.copy()
    states_goal_np = states_goal_np_inp.copy()

    is_solved_all: np.ndarray = np.zeros(states_curr_np.shape[0], dtype=bool)
    num_steps_all: np.ndarray = np.zeros(states_curr_np.shape[0])

    for _ in range(max_solve_steps):
        is_solved_all = (100 * np.mean(states_curr_np == states_goal_np, axis=1)) >= per_eq_tol

        idxs_unsolved = np.where(is_solved_all == 0)[0]
        if idxs_unsolved.shape[0] == 0:
            break

        states_curr_unsolved_np = states_curr_np[idxs_unsolved]
        states_goal_unsolved_np = states_goal_np[idxs_unsolved]

        if isinstance(dqn, FunctionType):
            assert isinstance(
                env_model,
                FunctionType), "If dqn is FunctionType then env_model must also be FunctionType"
            qvals_np = dqn(states_curr_unsolved_np, states_goal_unsolved_np)
            actions_np = qvals_np.argmin(axis=1)

            states_next_np = env_model(states_curr_unsolved_np, actions_np).round()
        else:
            states_curr_unsolved = (torch.tensor(states_curr_unsolved_np,
                                                 device=device).float().detach())
            states_goal_unsolved = (torch.tensor(states_goal_unsolved_np,
                                                 device=device).float().detach())
            qvals = dqn(states_curr_unsolved, states_goal_unsolved).detach()
            actions = torch.argmin(qvals, dim=1).float().detach()

            states_next = env_model(states_curr_unsolved, actions).round().detach()
            states_next_np = states_next.cpu().data.numpy()

        states_curr_np[idxs_unsolved] = states_next_np
        num_steps_all[idxs_unsolved] = num_steps_all[idxs_unsolved] + 1

    return is_solved_all, num_steps_all


def gbfs_test(
    states_offline: np.ndarray,
    num_states: int,
    dqn: nn.Module,
    env_model: nn.Module,
    num_actions: int,
    goal_steps_max: int,
    device: torch.device,
    max_solve_steps: int,
    per_eq_tol: float,
) -> float:
    """Test the GBFS algorithm on a set of states.

    Args:
        states_offline (np.ndarray): Offline states as numpy array.
        num_states (int): Number of states.
        dqn (nn.Module): The DQN model.
        env_model (nn.Module): The environment model.
        num_actions (int): Number of actions.
        goal_steps_max (int): Maximum number of goal steps.
        device (torch.device): Device to run the computations on.
        max_solve_steps (int): Maximum number of steps to solve.
        per_eq_tol (float): Percentage of elements that need to be equal to declare states equal.

    Returns:
        float: Percentage of solved states.
    """
    # init
    dqn.eval()
    env_model.eval()

    goal_steps_samp, num_states_per_goal_step = get_goal_steps(num_states, goal_steps_max)
    states_start_l, states_goal_l, goal_steps_l = generate_states(states_offline, goal_steps_samp,
                                                                  num_states_per_goal_step,
                                                                  num_actions, env_model, device)

    states_curr_np, states_goal_np, goal_steps = prepare_states(states_start_l, states_goal_l,
                                                                goal_steps_l)

    print(f"Solving {states_curr_np.shape[0]} states with GBFS")
    states_goal = torch.tensor(states_goal_np, device=device).float().detach()
    states_curr = torch.tensor(states_curr_np, device=device).float().detach()
    state_ctg_all: np.ndarray = (dqn(states_curr,
                                     states_goal).detach().min(dim=1)[0].cpu().data.numpy())

    is_solved_all, num_steps_all = gbfs(dqn, env_model, states_curr_np, states_goal_np, per_eq_tol,
                                        max_solve_steps, device)

    per_solved_all = 100 * float(sum(is_solved_all)) / float(len(is_solved_all))

    print(f"Greedy policy solved: {per_solved_all}")
    print_goal_step_stats(goal_steps, is_solved_all, num_steps_all, state_ctg_all)

    return per_solved_all


def setup_output_files(args: Namespace) -> Tuple[str, str]:
    """Setup output files for results and logs.

    Args:
        args (Namespace): Parsed arguments.

    Returns:
        Tuple[str, str]: Paths to results and output files.
    """
    results_file = f"{args.results_dir}/results.pkl"
    output_file = f"{args.results_dir}/output.txt"
    return results_file, output_file


def initialize_results() -> Dict[str, Any]:
    """Initialize the results dictionary.

    Returns:
        Dict[str, Any]: Initialized results dictionary.
    """
    return {
        "states": [],
        "solved": [],
        "num_steps": [],
        "times": [],
        "len_optimal_path": [],
        "is_optimal_path": [],
    }


def load_input_data(states_file: str) -> Dict[str, Any]:
    """Load input data from a file.

    Args:
        states_file (str): Path to the file containing states.

    Returns:
        Dict[str, Any]: Loaded input data.
    """
    with open(states_file, "rb") as f:
        return pickle.load(f)


def load_model_fn(args: Namespace, env: Environment, device: torch.device,
                  on_gpu: bool) -> nn.Module:
    """Load the environment model function.

    Args:
        args (Namespace): Parsed arguments.
        env (Environment): The environment.
        device (torch.device): Device to run the computations on.
        on_gpu (bool): Whether to use GPU.

    Returns:
        nn.Module: Loaded environment model function.
    """
    env_model_file = f"{args.env_model}/env_state_dict.pt"
    return nnet_utils.load_model_fn(env_model_file,
                                    device,
                                    on_gpu,
                                    env.get_env_nnet(),
                                    batch_size=args.nnet_batch_size)


def load_encoder(args: Namespace, env: Environment, device: torch.device) -> nn.Module:
    """Load the encoder model.

    Args:
        args (Namespace): Parsed arguments.
        env (Environment): The environment.
        device (torch.device): Device to run the computations on.

    Returns:
        nn.Module: Loaded encoder model.
    """
    encoder_file = f"{args.env_model}/encoder_state_dict.pt"
    encoder = nnet_utils.load_nnet(encoder_file, env.get_encoder(), device)
    encoder.eval()
    return encoder


def process_states(
    states: List[State],
    state_goals: List[State],
    env: Environment,
    encoder: nn.Module,
    heuristic_fn: nn.Module,
    model_fn: nn.Module,
    args: Namespace,
    results: Dict[str, Any],
    device: torch.device,
) -> None:
    """Process and solve the states.

    Args:
        states (List[State]): List of states.
        state_goals (List[State]): List of goal states.
        env (Environment): The environment.
        encoder (nn.Module): The encoder model.
        heuristic_fn (nn.Module): The heuristic function.
        model_fn (nn.Module): The environment model function.
        args (Namespace): Parsed arguments.
        results (Dict[str, Any]): Results dictionary.
        device (torch.device): Device to run the computations on.
    """
    start_idx = 0
    for state_idx in range(start_idx, len(states)):
        state = states[state_idx]
        state_goal = state_goals[state_idx]

        start_time = time.time()

        state_real = env.state_to_real([state])
        state_enc = encoder(torch.tensor(state_real, device=device).float())[1]
        state_enc_np = state_enc.cpu().data.numpy()

        if args.env == "sokoban":
            is_solved, num_steps = process_sokoban_state(state_goal, state_enc_np, env, encoder,
                                                         heuristic_fn, model_fn, args, device)
        else:
            is_solved, num_steps = process_other_state(state_goal, state_enc_np, env, encoder,
                                                       heuristic_fn, model_fn, args, device)

        solve_time = time.time() - start_time

        update_results(results, state, is_solved, num_steps, solve_time)

        print(f"State: {state_idx}, # Moves: {num_steps}, is_solved: {is_solved}, "
              f"Time: {solve_time:.2f}")


def process_sokoban_state(
    state_goal: SokobanState,
    state_enc_np: np.ndarray,
    env: Environment,
    encoder: nn.Module,
    heuristic_fn: nn.Module,
    model_fn: nn.Module,
    args: Namespace,
    device: torch.device,
) -> Tuple[bool, int]:
    """Process and solve a Sokoban state.

    Args:
        state_goal (SokobanState): The goal state.
        state_enc_np (np.ndarray): Encoded current state.
        env (Environment): The environment.
        encoder (nn.Module): The encoder model.
        heuristic_fn (nn.Module): The heuristic function.
        model_fn (nn.Module): The environment model function.
        args (Namespace): Parsed arguments.
        device (torch.device): Device to run the computations on.

    Returns:
        Tuple[bool, int]: Whether the state is solved and the number of steps taken.
    """
    blank_idxs = get_blank_indexes(state_goal)

    state_goals_i = generate_sokoban_goal_states(blank_idxs, state_goal)
    state_goals_i_real = env.state_to_real(state_goals_i)

    state_goals_i_enc = encoder(torch.tensor(state_goals_i_real, device=device).float())[1]
    state_goals_i_enc_np = state_goals_i_enc.cpu().data.numpy()

    states_enc_np = np.repeat(state_enc_np, state_goals_i_enc_np.shape[0], axis=0)
    is_solved_all, num_steps_all = gbfs(heuristic_fn, model_fn, states_enc_np,
                                        state_goals_i_enc_np, args.per_eq_tol, args.search_itrs,
                                        device)

    if np.max(is_solved_all):
        is_solved = True
        num_steps = np.min(num_steps_all[is_solved_all])
    else:
        is_solved = False
        num_steps = np.min(num_steps_all)

    return is_solved, num_steps


def process_other_state(
    state_goal: State,
    state_enc_np: np.ndarray,
    env: Environment,
    encoder: nn.Module,
    heuristic_fn: nn.Module,
    model_fn: nn.Module,
    args: Namespace,
    device: torch.device,
) -> Tuple[bool, int]:
    """Process and solve a non-Sokoban state.

    Args:
        state_goal (State): The goal state.
        state_enc_np (np.ndarray): Encoded current state.
        env (Environment): The environment.
        encoder (nn.Module): The encoder model.
        heuristic_fn (nn.Module): The heuristic function.
        model_fn (nn.Module): The environment model function.
        args (Namespace): Parsed arguments.
        device (torch.device): Device to run the computations on.

    Returns:
        Tuple[bool, int]: Whether the state is solved and the number of steps taken.
    """
    state_goal_real = env.state_to_real([state_goal])
    state_goal_enc = encoder(torch.tensor(state_goal_real, device=device).float())[1]
    state_goal_enc_np = state_goal_enc.cpu().data.numpy()

    is_solved_all, num_steps_all = gbfs(heuristic_fn, model_fn, state_enc_np, state_goal_enc_np,
                                        args.per_eq_tol, args.search_itrs, device)
    is_solved = is_solved_all[0]
    num_steps = num_steps_all[0]

    return is_solved, num_steps


def update_results(results: Dict[str, Any], state: State, is_solved: bool, num_steps: int,
                   solve_time: float) -> None:
    """Update the results dictionary with new data.

    Args:
        results (Dict[str, Any]): Results dictionary.
        state (State): The state.
        is_solved (bool): Whether the state is solved.
        num_steps (int): Number of steps taken.
        solve_time (float): Time taken to solve.
    """
    results["states"].append(state)
    results["solved"].append(is_solved)
    results["num_steps"].append(num_steps)
    results["times"].append(solve_time)

    if hasattr(state, "get_opt_path_len"):
        results["len_optimal_path"].append(state.get_opt_path_len())
        results["is_optimal_path"].append(num_steps <= state.get_opt_path_len())


def save_results(results: Dict[str, Any], results_file: str) -> None:
    """Save the results to a file.

    Args:
        results (Dict[str, Any]): Results dictionary.
        results_file (str): Path to the results file.
    """
    with open(results_file, "wb") as f:
        pickle.dump(results, f, protocol=-1)

    print_summary(results)


def print_summary(results: Dict[str, Any]) -> None:
    """Print a summary of the results.

    Args:
        results (Dict[str, Any]): Results dictionary.
    """
    opt_avg = _get_mean(results, "is_optimal_path") if "is_optimal_path" in results else 0

    print(f"Summary:\n"
          f"States Total: {len(results['states'])}, Solved Total: {np.sum(results['solved'])}\n"
          f"Means - SolnCost: {_get_mean(results, 'num_steps'):.2f}, "
          f"Solved: {100.0 * np.mean(results['solved']):.2f}%, "
          f"Optimal: {100.0 * opt_avg:.2f}%, "
          f"Time: {_get_mean(results, 'times'):.2f}")


def _get_mean(results: Dict[str, Any], key: str) -> float:
    """Calculate the mean of a specific key in the results dictionary.

    Args:
        results (Dict[str, Any]): Results dictionary.
        key (str): Key to calculate the mean for.

    Returns:
        float: Mean value.
    """
    vals = [x for x, solved in zip(results[key], results["solved"]) if solved]
    return np.mean(vals) if vals else 0


def get_goal_steps(num_states: int, goal_steps_max: int) -> Tuple[List[int], List[int]]:
    """Get goal steps and number of states per goal step.

    Args:
        num_states (int): Number of states.
        goal_steps_max (int): Maximum number of goal steps.

    Returns:
        Tuple[List[int], List[int]]: Goal steps and number of states per goal step.
    """
    goal_steps_samp = list(np.linspace(0, goal_steps_max, 30, dtype=int))
    num_states_per_goal_step = misc_utils.split_evenly(num_states, len(goal_steps_samp))
    return goal_steps_samp, num_states_per_goal_step


def generate_states(states_offline, goal_steps_samp, num_states_per_goal_step, num_actions,
                    env_model, device):
    states_start_l, states_goal_l, goal_steps_l = [], [], []

    for goal_step, goal_step_num_states in zip(goal_steps_samp, num_states_per_goal_step):
        if goal_step_num_states > 0:
            samp_idxs = np.random.randint(0, states_offline.shape[0], size=goal_step_num_states)
            goal_steps_i_l = [goal_step] * goal_step_num_states

            states_start_i = states_offline[samp_idxs]
            states_goal_i = random_walk(states_start_i, goal_steps_i_l, num_actions, env_model,
                                        device)

            states_start_l.append(states_start_i)
            states_goal_l.append(states_goal_i)
            goal_steps_l.extend(goal_steps_i_l)

    return states_start_l, states_goal_l, goal_steps_l


def prepare_states(states_start_l, states_goal_l,
                   goal_steps_l) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    states_curr_np = np.concatenate(states_start_l, axis=0)
    states_goal_np = np.concatenate(states_goal_l, axis=0)
    goal_steps = np.array(goal_steps_l)
    return states_curr_np, states_goal_np, goal_steps


def print_goal_step_stats(goal_steps, is_solved_all, num_steps_all, state_ctg_all):
    for goal_step in np.unique(goal_steps):
        step_idxs = np.where(goal_steps == goal_step)[0]
        if len(step_idxs) == 0:
            continue

        is_solved = is_solved_all[step_idxs]
        num_steps = num_steps_all[step_idxs]
        state_ctg = state_ctg_all[step_idxs]

        per_solved = 100 * float(sum(is_solved)) / float(len(is_solved))
        avg_solve_steps = np.mean(num_steps[is_solved]) if per_solved > 0.0 else 0.0

        print(f"Goal Steps: {goal_step}, "
              f"%Solved: {per_solved:.2f}, "
              f"avgSolveSteps: {avg_solve_steps:.2f}, "
              f"CTG Mean(Std/Min/Max): "
              f"{np.mean(state_ctg):.2f}"
              f"({np.std(state_ctg):.2f}/{np.min(state_ctg):.2f}/{np.max(state_ctg):.2f})")


def get_blank_indexes(state_goal: SokobanState) -> Set[Tuple[int, int]]:
    all_idxs = {(x, y) for x in range(10) for y in range(10)}

    wall_where = np.where(state_goal.walls)
    box_where = np.where(state_goal.boxes)

    wall_idxs = set(zip(wall_where[0], wall_where[1]))
    box_idxs = set(zip(box_where[0], box_where[1]))

    blocked_idxs = wall_idxs | box_idxs
    return all_idxs - blocked_idxs


def generate_sokoban_goal_states(blank_idxs, state_goal: SokobanState) -> List[SokobanState]:
    return [
        SokobanState(blank_idx, state_goal.boxes, state_goal.walls) for blank_idx in blank_idxs
    ]


def parse_arguments() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--states",
                        type=str,
                        required=True,
                        help="File containing states to solve")
    parser.add_argument("--env", type=str, required=True, help="Environment")
    parser.add_argument("--heur", type=str, required=True, help="Directory of heuristic function")
    parser.add_argument("--env_model", type=str, required=True, help="Directory of env model")
    parser.add_argument("--search_itrs", type=int, required=True, help="")
    parser.add_argument(
        "--per_eq_tol",
        type=float,
        required=True,
        help="Percent of latent state elements that need to be equal to declare equal")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--nnet_batch_size", type=int, default=None, help="")
    parser.add_argument("--debug", action="store_true", default=False, help="Set when debugging")
    return parser.parse_args()


def main():
    """Main function to run the GBFS test."""
    args = parse_arguments()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    results_file, output_file = setup_output_files(args)

    if not args.debug:
        sys.stdout = data_utils.Logger(output_file, "w")

    env = env_utils.get_environment(args.env)
    results = initialize_results()

    input_data = load_input_data(args.states)
    states, state_goals = input_data["states"], input_data["state_goals"]

    device, _, on_gpu = nnet_utils.get_device()
    heuristic_fn = nnet_utils.load_heuristic_fn(args.heur,
                                                device,
                                                on_gpu,
                                                env.get_dqn(),
                                                clip_zero=True,
                                                batch_size=args.nnet_batch_size)
    model_fn = load_model_fn(args, env, device, on_gpu)
    encoder = load_encoder(args, env, device)

    process_states(states, state_goals, env, encoder, heuristic_fn, model_fn, args, results,
                   device)

    save_results(results, results_file)


if __name__ == "__main__":
    main()
