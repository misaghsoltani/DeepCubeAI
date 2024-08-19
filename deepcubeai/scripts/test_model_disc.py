import os
import pickle
import time
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from deepcubeai.environments.environment_abstract import Environment
from deepcubeai.utils import env_utils, nnet_utils
from deepcubeai.utils.data_utils import print_args


def parse_arguments(parser: ArgumentParser) -> Dict[str, Any]:
    """Parses command-line arguments.

    Args:
        parser (ArgumentParser): The argument parser instance.

    Returns:
        Dict[str, Any]: A dictionary of parsed arguments.
    """
    parser.add_argument("--env", type=str, required=True, help="Environment")
    parser.add_argument("--data", type=str, required=True, help="Location of data")
    parser.add_argument("--env_dir",
                        type=str,
                        required=True,
                        help="Directory of environment model")
    parser.add_argument("--print_interval",
                        type=int,
                        default=1,
                        help="The interval of printing the info saving states to image files")

    args = parser.parse_args()
    args_dict: Dict[str, Any] = vars(args)
    print_args(args)

    return args_dict


def get_initial_state(state_episodes: List[np.ndarray], start_idxs: np.ndarray,
                      device: torch.device) -> torch.Tensor:
    """Gets the initial state tensor from state episodes and start indices.

    Args:
        state_episodes (List[np.ndarray]): List of state episodes.
        start_idxs (np.ndarray): Start indices for each episode.
        device (torch.device): The device to place the tensor on.

    Returns:
        torch.Tensor: The initial state tensor.
    """
    states_np = np.stack(
        [state_episode[idx] for state_episode, idx in zip(state_episodes, start_idxs)], axis=0)
    states = torch.tensor(states_np, device=device).float().contiguous()
    return states


def get_action(action_episodes: List[List[int]], start_idxs: np.ndarray, step: int,
               device: torch.device) -> torch.Tensor:
    """Gets the action tensor from action episodes, start indices, and step.

    Args:
        action_episodes (List[List[int]]): List of action episodes.
        start_idxs (np.ndarray): Start indices for each episode.
        step (int): The current step.
        device (torch.device): The device to place the tensor on.

    Returns:
        torch.Tensor: The action tensor.
    """
    actions_np = np.array(
        [action_episode[idx + step] for action_episode, idx in zip(action_episodes, start_idxs)])
    actions = torch.tensor(actions_np, device=device).float()
    return actions


def get_ground_truth_next_state(state_episodes: List[np.ndarray], start_idxs: np.ndarray,
                                step: int,
                                device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    """Gets the ground truth next state tensor and numpy array from state episodes, start indices,
        and step.

    Args:
        state_episodes (List[np.ndarray]): List of state episodes.
        start_idxs (np.ndarray): Start indices for each episode.
        step (int): The current step.
        device (torch.device): The device to place the tensor on.

    Returns:
        Tuple[torch.Tensor, np.ndarray]: The ground truth next state tensor and numpy array.
    """
    states_next_np = np.stack(
        [state_episode[idx + step + 1] for state_episode, idx in zip(state_episodes, start_idxs)],
        axis=0)
    states_next = torch.tensor(states_next_np, device=device).float().contiguous()
    return states_next, states_next_np


def calculate_metrics(
        encs_next_pred: torch.Tensor, encs_next: torch.Tensor, states_next_np: np.ndarray,
        states_next_pred_np: np.ndarray) -> Tuple[float, float, float, np.ndarray, float]:
    """Calculates various metrics for the predicted and ground truth encodings and states.

    Args:
        encs_next_pred (torch.Tensor): Predicted next encodings.
        encs_next (torch.Tensor): Ground truth next encodings.
        states_next_np (np.ndarray): Ground truth next states as numpy array.
        states_next_pred_np (np.ndarray): Predicted next states as numpy array.

    Returns:
        Tuple[float, float, float, np.ndarray, float]: The calculated metrics (eq, eq_bit,
            eq_bit_min, se, mse).
            - eq (float): Percentage of rows where all elements match.
            - eq_bit (float): Overall percentage of matching elements.
            - eq_bit_min (float): Minimum percentage of matching elements per row.
            - se (np.ndarray): Squared errors between ground truth and predicted states.
            - mse (float): Mean squared error between ground truth and predicted states.
    """
    eq_bits = torch.eq(encs_next_pred, torch.round(encs_next))
    eq = 100 * torch.all(eq_bits, dim=1).float().mean().cpu().data.numpy()
    eq_bit = 100 * eq_bits.float().mean().cpu().data.numpy()
    eq_bit_min = 100 * eq_bits.float().mean(dim=1).min().cpu().data.numpy()

    se = (states_next_np - states_next_pred_np)**2
    mse = float(np.mean(se))

    return eq, eq_bit, eq_bit_min, se, mse


def print_results(step: int, eq: float, eq_bit: float, eq_bit_min: float, mse: float,
                  all_match_count: int, not_matched_prints: int, save_dir: str,
                  states_next_np: np.ndarray, states_next_pred_np: np.ndarray) -> int:
    """Prints the results of the current step and saves images if required.

    Args:
        step (int): The current step.
        eq (float): Percentage of rows where all elements match.
        eq_bit (float): Overall percentage of matching elements.
        eq_bit_min (float): Minimum percentage of matching elements per row.
        mse (float): Mean squared error between ground truth and predicted states.
        all_match_count (int): The count of all matches so far.
        not_matched_prints (int): The count of not matched prints remaining.
        save_dir (str): The directory to save images.
        states_next_np (np.ndarray): Ground truth next states as numpy array.
        states_next_pred_np (np.ndarray): Predicted next states as numpy array.

    Returns:
        int: The updated count of not matched prints remaining.
    """
    print(f"step: {step}, eq_bit: {eq_bit:.2f}%, eq_bit_min: {eq_bit_min:.2f}%, "
          f"eq: {eq:.2f}%, mse: {mse:.2E}, all (so far/this step): "
          f"{all_match_count / (step + 1):.2f}/{int(eq == 100.0)}")
    not_matched_prints -= 1

    if save_dir is not None:
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
        fig.savefig(f"{save_dir}/model_test_disc_{step}.jpg")
        # pickle.dump(se_all, open(f"mse_test_disc_{step}.pkl", "wb"), protocol=-1)
        # pickle.dump(eq_l, open(f"eq_test_disc_{step}.pkl", "wb"), protocol=-1)
        plt.show(block=True)
        plt.close()

    return not_matched_prints


def step_model(encoder: nn.Module,
               env_model: nn.Module,
               dec_model: nn.Module,
               state_episodes: List[np.ndarray],
               action_episodes: List[List[int]],
               start_idxs: np.ndarray,
               device: torch.device,
               num_steps: int,
               print_interval: int = None,
               save_dir: str = None) -> List[List[float]]:
    """Steps through the model for a given number of steps and calculates metrics.

    Args:
        encoder (nn.Module): The encoder model.
        env_model (nn.Module): The environment model.
        dec_model (nn.Module): The decoder model.
        state_episodes (List[np.ndarray]): List of state episodes.
        action_episodes (List[List[int]]): List of action episodes.
        start_idxs (np.ndarray): Start indices for each episode.
        device (torch.device): The device to place the tensors on.
        num_steps (int): The number of steps to run.
        print_interval (int, optional): The interval for printing results and saving images.
            Defaults to None.
        save_dir (str, optional): The directory to save images. Defaults to None.

    Returns:
        List[List[float]]: The squared errors for each step.
    """
    states = get_initial_state(state_episodes, start_idxs, device)
    _, encs = encoder(states)

    all_match_count = 0
    eq_bit_min_all = np.inf
    se_all = []
    eq_l = []
    not_matched_prints = 5

    for step in range(num_steps):
        actions = get_action(action_episodes, start_idxs, step, device)
        encs_next_pred = env_model(encs, actions)
        encs_next_pred = torch.round(encs_next_pred)

        states_next, states_next_np = get_ground_truth_next_state(state_episodes, start_idxs, step,
                                                                  device)
        _, encs_next = encoder(states_next)
        states_next_pred_np = dec_model(encs_next_pred).detach().cpu().data.numpy()

        eq, eq_bit, eq_bit_min, se, mse = calculate_metrics(encs_next_pred, encs_next,
                                                            states_next_np, states_next_pred_np)
        eq_l.append(eq)
        se_all.append(list(se.mean(axis=(1, 2, 3))))

        all_match = eq == 100.0
        encs = encs_next_pred.detach()
        all_match_count += int(all_match)
        eq_bit_min_all = min(eq_bit_min_all, eq_bit_min)

        should_print = (print_interval is not None) and ((step == 0) or
                                                         (step % print_interval == 0) or
                                                         (step == num_steps - 1) or
                                                         ((not all_match) and
                                                          (not_matched_prints > 0)))

        if should_print:
            not_matched_prints = print_results(step, eq, eq_bit, eq_bit_min, mse, all_match_count,
                                               not_matched_prints, save_dir, states_next_np,
                                               states_next_pred_np)

    print(f"eq_bit_min (all): {eq_bit_min_all:.2f}%")
    print(f"{all_match_count} out of {num_steps} have all match")
    return se_all


def load_data(data_path: str) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Loads state and action episodes from a specified file.

    Args:
        data_path (str): Path to the data file.

    Returns:
        Tuple[List[np.ndarray], List[List[int]]]: Loaded state and action episodes.
    """
    with open(data_path, "rb") as data_file:
        state_episodes, action_episodes = pickle.load(data_file)
    return state_episodes, action_episodes


def setup_environment(
        args_dict: Dict[str, Any]) -> Tuple[torch.device, nn.Module, nn.Module, nn.Module]:
    """Sets up the environment and loads the models.

    Args:
        args_dict (Dict[str, Any]): Dictionary of parsed arguments.

    Returns:
        Tuple[torch.device, nn.Module, nn.Module, nn.Module]: The device, encoder model,
            environment model, and decoder model.
    """
    env: Environment = env_utils.get_environment(args_dict["env"])
    device, _, _ = nnet_utils.get_device()
    enc_model = nnet_utils.load_nnet(f"{args_dict['env_dir']}/encoder_state_dict.pt",
                                     env.get_encoder())
    env_model = nnet_utils.load_nnet(f"{args_dict['env_dir']}/env_state_dict.pt",
                                     env.get_env_nnet())
    dec_model = nnet_utils.load_nnet(f"{args_dict['env_dir']}/decoder_state_dict.pt",
                                     env.get_decoder())

    enc_model.to(device)
    env_model.to(device)
    dec_model.to(device)

    enc_model.eval()
    env_model.eval()
    dec_model.eval()

    return device, enc_model, env_model, dec_model


def main():
    """Main function to run the model testing."""
    parser = ArgumentParser()
    args_dict = parse_arguments(parser)

    print(f"HOST: {os.uname()[1]}")
    if "SLURM_JOB_ID" in os.environ:
        print(f"SLURM JOB ID: {os.environ['SLURM_JOB_ID']}")

    device, enc_model, env_model, dec_model = setup_environment(args_dict)

    print("Loading data ...")
    start_time = time.time()
    state_episodes, action_episodes = load_data(args_dict["data"])
    num_steps = len(action_episodes[0])
    start_idxs = np.zeros(len(state_episodes), dtype=int)

    print(f"{len(state_episodes)} episodes")
    print(f"Data load time: {time.time() - start_time}")

    save_dir = f"{args_dict['env_dir']}/model_test_pics"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Test images will be saved to '{save_dir}'")

    print(f"{len(state_episodes)} episodes, {num_steps} steps")
    step_model(enc_model, env_model, dec_model, state_episodes, action_episodes, start_idxs,
               device, num_steps, args_dict["print_interval"], save_dir)

    print("Done")


if __name__ == "__main__":
    main()
