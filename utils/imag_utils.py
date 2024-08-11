from typing import List, Tuple

import numpy as np
import torch
from torch import nn


def random_walk_traj(states_np_inp: np.ndarray, num_steps: int, num_actions: int,
                     env_model: nn.Module, device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a random walk trajectory for a given number of steps and actions.

    Args:
        states_np_inp (np.ndarray): Initial states as a NumPy array.
        num_steps (int): Number of steps to simulate.
        num_actions (int): Number of possible actions.
        env_model (nn.Module): The environment model to predict next states.
        device: The device (CPU or GPU) to perform computations on.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Final states and the trajectory of states.
    """
    states_np = states_np_inp.copy()
    states_traj_np = np.zeros((states_np.shape[0], num_steps + 1, states_np.shape[1]))
    states_to_move = torch.tensor(states_np, device=device).float().detach()

    for step_num in range(num_steps):
        states_traj_np[:, step_num, :] = states_to_move.cpu().data.numpy()

        actions_np = np.random.randint(0, num_actions, size=states_np.shape[0])
        actions = torch.tensor(actions_np, device=device).float().detach()

        states_to_move = env_model(states_to_move, actions).round().detach()

    states_np = states_to_move.cpu().data.numpy()
    states_traj_np[:, -1, :] = states_np

    return states_np, states_traj_np


def random_walk(states_np_inp: np.ndarray, num_steps_l: List[int], num_actions: int,
                env_model: nn.Module, device) -> np.ndarray:
    """
    Performs a random walk for a list of step counts and actions.

    Args:
        states_np_inp (np.ndarray): Initial states as a NumPy array.
        num_steps_l (List[int]): List of step counts for each state.
        num_actions (int): Number of possible actions.
        env_model (nn.Module): The environment model to predict next states.
        device: The device (CPU or GPU) to perform computations on.

    Returns:
        np.ndarray: Final states after the random walk.
    """
    # initialize
    num_steps_max: int = max(num_steps_l)
    num_steps: np.ndarray = np.array(num_steps_l)
    num_steps_curr: np.ndarray = np.array(num_steps_l)
    states_np = states_np_inp.copy()

    states_to_move = torch.tensor(states_np, device=device).float().detach()

    for step_num in range(num_steps_max):
        # get actions
        actions_np = np.random.randint(0, num_actions, size=states_to_move.shape[0])
        actions = torch.tensor(actions_np, device=device).float().detach()

        # get next states
        states_to_move = env_model(states_to_move, actions).round().detach()

        # record goal states
        end_step_mask = num_steps == (step_num + 1)
        end_step_mask_curr = num_steps_curr == (step_num + 1)
        states_np[end_step_mask] = states_to_move[end_step_mask_curr].to(
            torch.uint8).cpu().data.numpy()

        # get only states that have not reached goal state
        move_mask_curr = num_steps_curr > (step_num + 1)
        states_to_move = states_to_move[move_mask_curr]
        num_steps_curr = num_steps_curr[move_mask_curr]

    return states_np
