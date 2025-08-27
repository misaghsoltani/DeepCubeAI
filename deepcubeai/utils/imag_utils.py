from __future__ import annotations

import numpy as np
from numpy import float32, intp
from numpy.typing import NDArray
import torch
from torch import nn


def random_walk_traj(
    states_np_inp: NDArray[float32], num_steps: int, num_actions: int, env_model: nn.Module, device: torch.device
) -> tuple[NDArray[float32], NDArray[float32]]:
    """Generates a random walk trajectory for a given number of steps and actions.

    Args:
        states_np_inp (np.NDArray): Initial states as a NumPy array.
        num_steps (int): Number of steps to simulate.
        num_actions (int): Number of possible actions.
        env_model (nn.Module): The environment model to predict next states.
        device: The device (CPU or GPU) to perform computations on.

    Returns:
        tuple[NDArray, NDArray]: Final states and the trajectory of states.
    """
    states_np: NDArray[float32] = np.asarray(states_np_inp.copy(), dtype=float32)
    states_traj_np: NDArray[float32] = np.zeros((states_np.shape[0], num_steps + 1, states_np.shape[1]), dtype=float32)
    states_to_move = torch.tensor(states_np, device=device).float().detach()

    for step_num in range(num_steps):
        states_traj_np[:, step_num, :] = states_to_move.cpu().data.numpy()

        actions_np = np.random.randint(0, num_actions, size=states_np.shape[0])
        actions = torch.tensor(actions_np, device=device).float().detach()

        states_to_move = env_model(states_to_move, actions).round().detach()

    states_np = np.asarray(states_to_move.cpu().data.numpy(), dtype=float32)
    states_traj_np[:, -1, :] = states_np

    return states_np, states_traj_np


def random_walk(
    states_np_inp: NDArray[float32],
    num_steps_l: list[int],
    num_actions: int,
    env_model: nn.Module,
    device: torch.device,
) -> NDArray[float32]:
    """Performs a random walk for a list of step counts and actions.

    Args:
        states_np_inp (np.NDArray): Initial states as a NumPy array.
        num_steps_l (list[int]): List of step counts for each state.
        num_actions (int): Number of possible actions.
        env_model (nn.Module): The environment model to predict next states.
        device: The device (CPU or GPU) to perform computations on.

    Returns:
        NDArray: Final states after the random walk.
    """
    # initialize
    num_steps_max: int = max(num_steps_l)
    num_steps: NDArray[intp] = np.array(num_steps_l)
    num_steps_curr: NDArray[intp] = np.array(num_steps_l)
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
        states_np[end_step_mask] = states_to_move[end_step_mask_curr].to(torch.uint8).cpu().data.numpy()

        # get only states that have not reached goal state
        move_mask_curr = num_steps_curr > (step_num + 1)
        states_to_move = states_to_move[move_mask_curr]
        num_steps_curr = num_steps_curr[move_mask_curr]

    return states_np
