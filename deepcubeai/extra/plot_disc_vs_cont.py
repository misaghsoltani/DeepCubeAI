import os
import pickle
import sys
import time
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from deepcubeai.environments.environment_abstract import Environment
from deepcubeai.scripts.test_model_cont import step_model as step_model_cont
from deepcubeai.scripts.test_model_disc import step_model as step_model_disc
from deepcubeai.utils import data_utils, env_utils, nnet_utils
from deepcubeai.utils.data_utils import print_args


def generate_file_name(save_dir: str, prefix: str) -> str:
    """Generates a unique file name with a given prefix in the specified directory.

    Args:
        save_dir (str): The directory where the file will be saved.
        prefix (str): The prefix for the file name.

    Returns:
        str: The generated file name with the prefix.
    """
    existing_files = [file for file in os.listdir(save_dir) if file.startswith(prefix)]
    if existing_files:
        max_suffix = max(int(file.split("_")[-1].split(".")[0]) for file in existing_files)
        suffix = max_suffix + 1
    else:
        suffix = 1
    return f"{prefix}{suffix}"


def calculate_statistics(se_cont_l: List[List[float]], se_disc_l: List[List[float]]) -> str:
    """Calculates and returns statistical information for continuous and discrete errors.

    Args:
        se_cont_l (List[List[float]]): List of squared errors for the continuous model.
        se_disc_l (List[List[float]]): List of squared errors for the discrete model.

    Returns:
        str: A formatted string containing statistical information.
    """

    def calculate_metrics(se: np.ndarray) -> Dict[str, float]:
        """Calculates various statistical metrics for the given squared errors.

        Args:
            se (np.ndarray): Array of squared errors.

        Returns:
            Dict[str, float]: Dictionary containing statistical metrics.
        """
        return {
            "mean": np.mean(se),
            "min": np.min(se),
            "max": np.max(se),
            "std": np.std(se),
            "var": np.var(se),
            "median": np.median(se),
            "q1": np.percentile(se, 25),
            "q3": np.percentile(se, 75),
            "iqr": np.percentile(se, 75) - np.percentile(se, 25),
            "range": np.max(se) - np.min(se),
        }

    se_cont = np.array(se_cont_l)
    se_disc = np.array(se_disc_l)

    cont_metrics = calculate_metrics(se_cont)
    disc_metrics = calculate_metrics(se_disc)

    info = (f"\nDiscrete Model:\n"
            f"-----\n"
            f"Overall Mean: {disc_metrics['mean']:.3e}\n"
            f"Overall Min: {disc_metrics['min']:.3e}\n"
            f"Overall Max: {disc_metrics['max']:.3e}\n"
            f"Std Dev: {disc_metrics['std']:.3e}\n"
            f"Variance: {disc_metrics['var']:.3e}\n"
            f"Median: {disc_metrics['median']:.3e}\n"
            f"Q1: {disc_metrics['q1']:.3e}\n"
            f"Q3: {disc_metrics['q3']:.3e}\n"
            f"IQR: {disc_metrics['iqr']:.3e}\n"
            f"Range: {disc_metrics['range']:.3e}\n"
            f"\nContinuous Model:\n"
            f"-----\n"
            f"Overall Mean: {cont_metrics['mean']:.3e}\n"
            f"Overall Min: {cont_metrics['min']:.3e}\n"
            f"Overall Max: {cont_metrics['max']:.3e}\n"
            f"Std Dev: {cont_metrics['std']:.3e}\n"
            f"Variance: {cont_metrics['var']:.3e}\n"
            f"Median: {cont_metrics['median']:.3e}\n"
            f"Q1: {cont_metrics['q1']:.3e}\n"
            f"Q3: {cont_metrics['q3']:.3e}\n"
            f"IQR: {cont_metrics['iqr']:.3e}\n"
            f"Range: {cont_metrics['range']:.3e}\n")

    return info


def parse_arguments() -> Namespace:
    """Parses command-line arguments.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment")
    parser.add_argument("--model_test_data", type=str, required=True, help="Location of data")
    parser.add_argument("--env_model_dir_disc",
                        type=str,
                        required=True,
                        help="Directory of environment model")
    parser.add_argument("--env_model_dir_cont",
                        type=str,
                        required=True,
                        help="Directory of environment model")
    parser.add_argument("--num_episodes",
                        type=int,
                        default=-1,
                        help="Number of episodes to be used")
    parser.add_argument("--num_steps", type=int, default=-1, help="Number of steps to be used")
    parser.add_argument("--print_interval",
                        type=int,
                        default=1,
                        help="The interval of printing the info")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the image")
    parser.add_argument("--save_pdf", action="store_true", default=True, help="Save plot as PDF")
    return parser.parse_args()


def load_data(model_test_data: str) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Loads test data from a specified file.

    Args:
        model_test_data (str): Path to the test data file.

    Returns:
        Tuple[List[np.ndarray], List[List[int]]]: Loaded state and action episodes.
    """
    print("Loading data...")
    start_time = time.time()
    with open(model_test_data, "rb") as file:
        state_episodes, action_episodes = pickle.load(file)
    print(f"Data load time: {time.time() - start_time}\n")
    return state_episodes, action_episodes


def setup_environment(
        args: Namespace) -> Tuple[Environment, List[np.ndarray], List[List[int]], int, int, int]:
    """Sets up the environment and loads the state and action episodes.

    Args:
        args (Namespace): Parsed command-line arguments.

    Returns:
        Tuple[Environment, List[np.ndarray], List[List[int]], int, int, int]: Environment,
            state episodes, action episodes, number of state episodes, number of steps, and
            starting episode index.
    """
    env = env_utils.get_environment(args.env)
    state_episodes, action_episodes = load_data(args.model_test_data)
    num_state_episodes = len(state_episodes)
    num_eps = args.num_episodes if args.num_episodes > 0 else num_state_episodes

    start_ep = 0
    if num_eps < num_state_episodes:
        start_ep = randint(0, num_state_episodes - num_eps - 1)
        state_episodes = state_episodes[start_ep:start_ep + num_eps]
        action_episodes = action_episodes[start_ep:start_ep + num_eps]
        num_state_episodes = len(state_episodes)

    # number of steps
    len_steps = len(action_episodes[0])  # TODO assuming all episodes the same len
    num_steps = (args.num_steps if
                 (args.num_steps > 0 and args.num_steps < len_steps) else len_steps)

    if num_steps < len_steps:
        state_episodes = [x[0:num_steps + 1] for x in state_episodes]
        action_episodes = [x[0:num_steps] for x in action_episodes]

    return env, state_episodes, action_episodes, num_state_episodes, num_steps, start_ep


def setup_save_paths(save_dir: str,
                     env_name: str,
                     num_state_episodes: int,
                     num_steps: int,
                     save_pdf: bool = False) -> Tuple[str, str, str]:
    """Sets up the save paths for plots and information.

    Args:
        save_dir (str): The directory where the plot will be saved to.
        env_name (str): The name of the environment.
        num_state_episodes (int): Number of state episodes.
        num_steps (int): Number of steps.
        save_pdf (bool, optional): Whether to save the plot as a PDF. Defaults to False.

    Returns:
        Tuple[str, str, str]: Save directory, plot save path, and text save path.
    """
    save_dir = os.getcwd() if save_dir is None else save_dir
    save_dir = os.path.join(save_dir, "plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fname_prefix = f"{env_name}_mse_{num_state_episodes}eps_{num_steps}steps_"
    file_name = generate_file_name(save_dir, fname_prefix)
    file_extension = "pdf" if save_pdf else "png"
    plot_save_path = os.path.join(save_dir, f"{file_name}.{file_extension}")
    txt_save_path = os.path.join(save_dir, f"{file_name}.txt")
    print(f"Plot will be saved to '{os.path.abspath(plot_save_path)}'")
    print(f"Information will be saved to '{os.path.abspath(txt_save_path)}'")

    return save_dir, plot_save_path, txt_save_path


def load_models(env: Environment, args: Namespace,
                device: torch.device) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """Loads the neural network models for the environment.

    Args:
        env (Environment): The environment object.
        args (Namespace): Parsed command-line arguments.
        device (torch.device): The device to load the models onto.

    Returns:
        Tuple[nn.Module, nn.Module, nn.Module, nn.Module]: Encoder model, environment model,
            decoder model, and continuous environment model.
    """
    enc_file = f"{args.env_model_dir_disc}/encoder_state_dict.pt"
    env_file = f"{args.env_model_dir_disc}/env_state_dict.pt"
    dec_file = f"{args.env_model_dir_disc}/decoder_state_dict.pt"

    enc_model = nnet_utils.load_nnet(enc_file, env.get_encoder())
    env_model = nnet_utils.load_nnet(env_file, env.get_env_nnet())
    dec_model = nnet_utils.load_nnet(dec_file, env.get_decoder())

    env_file_cont = f"{args.env_model_dir_cont}/model_state_dict.pt"
    env_model_cont = nnet_utils.load_nnet(env_file_cont, env.get_env_nnet_cont())

    for nnet in [enc_model, env_model, dec_model, env_model_cont]:
        nnet.to(device)
        nnet.eval()

    return enc_model, env_model, dec_model, env_model_cont


def plot_results(se_disc_l: List[List[float]], se_cont_l: List[List[float]], num_steps: int,
                 plot_save_path: str) -> None:
    """Plots the results of the discrete and continuous models.

    Args:
        se_disc_l (List[List[float]]): List of squared errors for the discrete model.
        se_cont_l (List[List[float]]): List of squared errors for the continuous model.
        num_steps (int): Number of steps.
        plot_save_path (str): Path to save the plot.
    """
    plt.ion()
    x_vals = list(range(1, num_steps + 1))
    se_mean = [np.mean(x) for x in se_disc_l]
    plt.errorbar(x_vals, se_mean, label="Discrete")

    se_mean = [np.mean(x) for x in se_cont_l]
    plt.plot(x_vals, se_mean, label="Continuous")

    plt.legend()
    plt.savefig(plot_save_path)
    plt.show(block=True)


def main():
    """Main function to run the model testing and plotting."""
    args = parse_arguments()
    env, state_episodes, action_episodes, num_state_episodes, num_steps, start_ep = (
        setup_environment(args))
    _, plot_save_path, txt_save_path = setup_save_paths(args.save_dir, args.env,
                                                        num_state_episodes, num_steps,
                                                        args.save_pdf)

    sys.stdout = data_utils.Logger(txt_save_path, "a")
    print_args(args)
    print(f"Episodes {start_ep}-{start_ep + num_state_episodes} ({num_state_episodes} episodes, "
          f"{num_steps} steps)")

    start_idxs = np.zeros(num_state_episodes, dtype=int)
    device, _, _ = nnet_utils.get_device()
    enc_model, env_model, dec_model, env_model_cont = load_models(env, args, device)

    print("\nTesting the discrete model:")
    se_disc_l = step_model_disc(enc_model, env_model, dec_model, state_episodes, action_episodes,
                                start_idxs, device, num_steps, args.print_interval)

    print("\nTesting the continuous model:")
    se_cont_l = step_model_cont(env_model_cont, state_episodes, action_episodes, start_idxs,
                                device, num_steps, args.print_interval)

    result_info = calculate_statistics(se_cont_l, se_disc_l)
    print(result_info)

    plot_results(se_disc_l, se_cont_l, num_steps, plot_save_path)
    print(f"Plot saved to '{os.path.abspath(plot_save_path)}'")
    print(f"Information saved to '{os.path.abspath(txt_save_path)}'")


if __name__ == "__main__":
    main()
