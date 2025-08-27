from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
import os
import pickle
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import float32, intp
from numpy.typing import NDArray
import torch
from torch import nn

from deepcubeai.environments.environment_abstract import Environment
from deepcubeai.utils import env_utils, nnet_utils
from deepcubeai.utils.data_utils import print_args


def parse_arguments(parser: ArgumentParser) -> dict[str, Any]:
    """Parses command-line arguments.

    Args:
        parser (ArgumentParser): The argument parser instance.

    Returns:
        dict[str, Any]: A dictionary of parsed arguments.
    """
    parser.add_argument("--env", type=str, required=True, help="Environment")
    parser.add_argument("--data", type=str, required=True, help="Location of data")
    parser.add_argument("--env_dir", type=str, required=True, help="Directory of environment model")
    parser.add_argument(
        "--print_interval",
        type=int,
        default=1,
        help="The interval of printing the info and saving states to image files",
    )

    args = parser.parse_args()
    args_dict: dict[str, Any] = vars(args)
    print_args(args)

    return args_dict


@dataclass(frozen=True, slots=True)
class TestModelContConfig:
    """Config for continuous model testing."""

    env: str
    data: str
    env_dir: str
    print_interval: int = 1

    @staticmethod
    def from_json(path: str | os.PathLike[str]) -> TestModelContConfig:
        """Load config from JSON path."""
        with open(path, "rb") as f:
            raw: dict[str, Any] = json.loads(f.read())
        return TestModelContConfig(**raw)


def get_initial_states(
    state_episodes: list[NDArray[float32]], start_idxs: NDArray[intp], device: torch.device
) -> torch.Tensor:
    """Gets the initial states from the state episodes.

    Args:
        state_episodes (list[NDArray]): List of state episodes.
        start_idxs (np.NDArray): Array of start indices.
        device (torch.device): The device to use for tensor operations.

    Returns:
        torch.Tensor: Tensor of initial states.
    """
    states_np: NDArray[float32] = np.stack(
        [state_episode[idx] for state_episode, idx in zip(state_episodes, start_idxs, strict=True)], axis=0
    )
    return torch.tensor(states_np, device=device).float().contiguous()


def get_actions(
    action_episodes: list[list[int]], start_idxs: NDArray[intp], step: int, device: torch.device
) -> torch.Tensor:
    """Gets the actions for the given step from the action episodes.

    Args:
        action_episodes (list[list[int]]): List of action episodes.
        start_idxs (np.NDArray): Array of start indices.
        step (int): The current step.
        device (torch.device): The device to use for tensor operations.

    Returns:
        torch.Tensor: Tensor of actions.
    """
    actions_np: NDArray[intp] = np.array([
        action_episode[idx + step] for action_episode, idx in zip(action_episodes, start_idxs, strict=True)
    ])
    return torch.tensor(actions_np, device=device).float()


def get_next_states(state_episodes: list[NDArray[float32]], start_idxs: NDArray[intp], step: int) -> NDArray[float32]:
    """Gets the next states for the given step from the state episodes.

    Args:
        state_episodes (list[NDArray]): List of state episodes.
        start_idxs (np.NDArray): Array of start indices.
        step (int): The current step.

    Returns:
        NDArray: Array of next states.
    """
    return np.stack(
        [state_episode[idx + step + 1] for state_episode, idx in zip(state_episodes, start_idxs, strict=True)], axis=0
    )


def calculate_se(states_next_np: NDArray[float32], states_next_pred_np: NDArray[float32]) -> tuple[list[float], float]:
    """Calculates the squared error between the actual and predicted next states.

    Args:
        states_next_np (np.NDArray): Array of actual next states.
        states_next_pred_np (np.NDArray): Array of predicted next states.

    Returns:
        tuple[list[float], float]: List of mean squared errors for each state and the overall
            mean squared error.
    """
    se = (states_next_np - states_next_pred_np) ** 2
    se_mean = list(se.mean(axis=(1, 2, 3)))
    mse = float(np.mean(se))
    return se_mean, mse


def plot_and_save_images(
    states_next_np: NDArray[float32], states_next_pred_np: NDArray[float32], step: int, save_dir: str
) -> None:
    """Plots and saves images of the actual and predicted next states.

    Args:
        states_next_np (np.NDArray): Array of actual next states.
        states_next_pred_np (np.NDArray): Array of predicted next states.
        step (int): The current step.
        save_dir (str): Directory to save the images.
    """
    fig = plt.figure(dpi=200)

    state_next_np = np.clip(states_next_np[0], 0, 1)
    state_next_pred_np = np.clip(states_next_pred_np[0], 0, 1)
    state_next_np = np.transpose(state_next_np, [1, 2, 0])
    state_next_pred_np = np.transpose(state_next_pred_np, [1, 2, 0])
    if state_next_np.shape[2] == 6:
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.imshow(state_next_np[:, :, :3])
        ax1.set_title("gt")
        ax2.imshow(state_next_np[:, :, 3:])
        ax2.set_title("gt")

        ax3.imshow(state_next_pred_np[:, :, :3])
        ax3.set_title("nnet")
        ax4.imshow(state_next_pred_np[:, :, 3:])
        ax4.set_title("nnet")
    else:
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)

        ax1.imshow(state_next_np)
        ax1.set_title("gt")
        ax2.imshow(state_next_pred_np)
        ax2.set_title("nnet")

    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.tight_layout()
    fig.savefig(f"{save_dir}/model_test_cont_{step}.jpg")

    plt.show(block=True)
    plt.close()


@torch.inference_mode()
def step_model(
    env_model: nn.Module,
    state_episodes: list[NDArray[float32]],
    action_episodes: list[list[int]],
    start_idxs: NDArray[intp],
    device: torch.device,
    num_steps: int,
    print_interval: int | None = None,
    save_dir: str | None = None,
) -> list[list[float]]:
    """Steps through the model for a given number of steps and calculates the squared error.

    Args:
        env_model (nn.Module): The environment model.
        state_episodes (list[NDArray]): List of state episodes.
        action_episodes (list[list[int]]): List of action episodes.
        start_idxs (np.NDArray): Array of start indices.
        device (torch.device): The device to use for tensor operations.
        num_steps (int): Number of steps to run the model.
        print_interval (int, optional): Interval for printing the mean squared error. Defaults
            to None.
        save_dir (str, optional): Directory to save the images. Defaults to None.

    Returns:
        list[list[float]]: List of mean squared errors for each step.
    """
    states = get_initial_states(state_episodes, start_idxs, device)

    se_all: list[list[float]] = []
    mse_sum: float = 0.0
    for step in range(num_steps):
        actions = get_actions(action_episodes, start_idxs, step, device)
        states_next_pred = env_model(states, actions)
        states_next_pred_np = states_next_pred.cpu().data.numpy()
        states_next_np = get_next_states(state_episodes, start_idxs, step)

        se_mean, mse = calculate_se(states_next_np, states_next_pred_np)
        se_all.append(se_mean)
        mse_sum += mse

        if (print_interval is not None) and ((step == 0) or (step % print_interval == 0) or (step == num_steps - 1)):
            print(f"step: {step}, mse(so far/this step): {mse_sum / (step + 1):.2e}/{mse:.2e}")

            if save_dir is not None:
                plot_and_save_images(states_next_np, states_next_pred_np, step, save_dir)

        states = states_next_pred.detach()

    return se_all


@torch.inference_mode()
def main() -> None:
    """Main function to run the model testing."""
    parser: ArgumentParser = ArgumentParser()
    args_dict: dict[str, Any] = parse_arguments(parser)

    print(f"HOST: {os.uname()[1]}")

    if "SLURM_JOB_ID" in os.environ:
        print(f"SLURM JOB ID: {os.environ['SLURM_JOB_ID']}")

    env: Environment = env_utils.get_environment(args_dict["env"])

    print("Loading data ...")
    start_time = time.time()
    state_episodes: list[NDArray[float32]]

    with open(args_dict["data"], "rb") as file:
        state_episodes, action_episodes = pickle.load(file)

    start_idxs: NDArray[intp] = np.zeros(len(state_episodes), dtype=int)
    num_steps: int = len(action_episodes[0])  # TODO assuming all episodes the same len

    print(f"{len(state_episodes)} episodes")
    print(f"{num_steps} steps")
    print(f"Data load time: {(time.time() - start_time)}")

    device, devices, on_gpu = nnet_utils.get_device()
    print(f"device: {device}, devices: {devices}, on_gpu: {on_gpu}")

    # load nnet
    env_file = f"{args_dict['env_dir']}/model_state_dict.pt"

    env_model = nnet_utils.load_nnet(env_file, env.get_env_nnet_cont())
    env_model.to(device)
    env_model.eval()

    save_dir = f"{args_dict['env_dir']}/model_test_pics"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Test images will be saved to '{save_dir}'")

    print(f"{len(state_episodes)} episodes, {num_steps} steps")
    step_model(
        env_model, state_episodes, action_episodes, start_idxs, device, num_steps, args_dict["print_interval"], save_dir
    )

    print("Done")


@torch.inference_mode()
def run_test_model_cont(cfg: TestModelContConfig) -> None:
    """Programmatic entrypoint for testing continuous model."""
    env: Environment = env_utils.get_environment(cfg.env)
    device, _, _ = nnet_utils.get_device()
    with open(cfg.data, "rb") as f:
        state_episodes, action_episodes = pickle.load(f)
    start_idxs = np.zeros(len(state_episodes), dtype=int)
    num_steps = len(action_episodes[0])
    env_model = nnet_utils.load_nnet(f"{cfg.env_dir}/model_state_dict.pt", env.get_env_nnet_cont())
    env_model.to(device)
    env_model.eval()
    save_dir = f"{cfg.env_dir}/model_test_pics"
    os.makedirs(save_dir, exist_ok=True)
    step_model(env_model, state_episodes, action_episodes, start_idxs, device, num_steps, cfg.print_interval, save_dir)


if __name__ == "__main__":
    main()
