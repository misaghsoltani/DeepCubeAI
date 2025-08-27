from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np
from numpy import float32, float64, intp
from numpy.typing import NDArray
import torch
from torch import Tensor, nn, optim
from torch.optim.optimizer import Optimizer

from deepcubeai.environments.environment_abstract import Environment
from deepcubeai.utils import data_utils, env_utils, nnet_utils
from deepcubeai.utils.data_utils import print_args


def parse_arguments(parser: ArgumentParser) -> dict[str, Any]:
    """Parses command-line arguments.

    Args:
        parser (ArgumentParser): The argument parser instance.

    Returns:
        dict[str, Any]: A dictionary of parsed arguments.
    """
    # Environment
    parser.add_argument("--env", type=str, required=True, help="Environment")

    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Location of training data")
    parser.add_argument("--val_data", type=str, required=True, help="Location of validation data")

    # Debug
    parser.add_argument("--debug", action="store_true", default=False, help="")

    # Gradient Descent
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument(
        "--lr_d",
        type=float,
        default=0.9999993,
        help="Learning rate decay for every iteration. Learning rate is decayed according to: lr * (lr_d ^ itr)",
    )

    # Training
    parser.add_argument("--max_itrs", type=int, default=1000000, help="Maxmimum number of iterations")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--path_len_incr_itr", type=int, default=3000, help="Increment path length every x itrs")
    parser.add_argument("--num_steps", type=int, default=100, help="Maximum number of steps to predict")

    # model
    parser.add_argument("--nnet_name", type=str, required=True, help="Name of neural network")
    parser.add_argument("--save_dir", type=str, default="saved_env_models", help="Director to which to save model")

    # parse arguments
    args = parser.parse_args()
    args_dict: dict[str, Any] = vars(args)

    # make save directory
    train_dir: str = f"{args_dict['save_dir']}/{args_dict['nnet_name']}/"
    args_dict["train_dir"] = train_dir
    args_dict["nnet_model_dir"] = f"{args_dict['train_dir']}/"
    if not os.path.exists(args_dict["nnet_model_dir"]):
        os.makedirs(args_dict["nnet_model_dir"])

    args_dict["output_save_loc"] = f"{train_dir}/output.txt"

    # save args
    args_save_loc = f"{train_dir}/args.pkl"
    print(f"Saving arguments to {args_save_loc}")
    with open(args_save_loc, "wb") as f:
        pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Batch size: {args_dict['batch_size']}")

    return args_dict


@dataclass(frozen=True, slots=True)
class TrainEnvContConfig:
    """Configuration for continuous environment model training.

    Mirrors CLI flags to allow programmatic execution.
    """

    # Environment and data
    env: str
    train_data: str
    val_data: str

    # Training
    nnet_name: str
    save_dir: str = "saved_env_models"
    lr: float = 0.001
    lr_d: float = 0.9999993
    max_itrs: int = 1000000
    batch_size: int = 1000
    path_len_incr_itr: int = 3000
    num_steps: int = 100

    # Misc
    debug: bool = False

    @staticmethod
    def from_json(path: str | Path) -> TrainEnvContConfig:
        """Load config from a JSON file, ignoring unknown keys."""
        data = json.loads(Path(path).read_bytes())
        if not isinstance(data, dict):
            raise TypeError("Config JSON must be an object")
        allowed = set(TrainEnvContConfig.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in allowed}
        return TrainEnvContConfig(**filtered)


def run_with_argsd(args_dict: dict[str, Any]) -> None:
    """Run the existing training flow given a populated args dict."""
    if not args_dict["debug"] and not isinstance(sys.stdout, data_utils.Logger):
        sys.stdout = data_utils.Logger(args_dict["output_save_loc"], "a")

    print_args(args_dict)

    # environment
    env: Environment = env_utils.get_environment(args_dict["env"])

    # load nnet
    nnet_model: nn.Module
    start_itr: int
    nnet_model, start_itr = load_train_state(args_dict["train_dir"], args_dict["nnet_model_dir"], env)

    print(f"Starting iteration: {start_itr}, Max iteration: {args_dict['max_itrs']}")

    if args_dict["max_itrs"] <= start_itr:
        print("Starting iteration >= Max iteration. Skipping training for these iterations.")
        return

    # get device
    device: torch.device
    devices: list[int]
    on_gpu: bool
    device, devices, on_gpu = nnet_utils.get_device()

    print(f"device: {device}, devices: {devices}, on_gpu: {on_gpu}")

    nnet_model.to(device)

    # load data
    print("Loading data ...")
    start_time = time.time()
    with open(args_dict["train_data"], "rb") as f:
        state_episodes_train, actions_train = pickle.load(f)
    with open(args_dict["val_data"], "rb") as f:
        state_episodes_val, actions_val = pickle.load(f)

    print(f"Data load time: {time.time() - start_time}")

    # train nnet
    train_nnet(
        nnet_model,
        state_episodes_train,
        actions_train,
        state_episodes_val,
        actions_val,
        args_dict["num_steps"],
        device,
        args_dict["batch_size"],
        args_dict["max_itrs"],
        start_itr,
        args_dict["lr"],
        args_dict["lr_d"],
        args_dict["nnet_model_dir"],
    )

    print("--------------\nDone\n")


def run_train_env_cont(cfg: TrainEnvContConfig) -> None:
    """Programmatic API to train continuous env model using this module's logic."""
    train_dir = f"{cfg.save_dir}/{cfg.nnet_name}/"
    nnet_model_dir = f"{train_dir}/"
    Path(nnet_model_dir).mkdir(parents=True, exist_ok=True)

    argsd: dict[str, Any] = {
        "env": cfg.env,
        "train_data": cfg.train_data,
        "val_data": cfg.val_data,
        "debug": cfg.debug,
        "lr": cfg.lr,
        "lr_d": cfg.lr_d,
        "max_itrs": cfg.max_itrs,
        "batch_size": cfg.batch_size,
        "path_len_incr_itr": cfg.path_len_incr_itr,
        "num_steps": cfg.num_steps,
        "nnet_name": cfg.nnet_name,
        "save_dir": cfg.save_dir,
        "train_dir": train_dir,
        "nnet_model_dir": nnet_model_dir,
        "output_save_loc": f"{train_dir}/output.txt",
    }

    # Save cfg snapshot similar to CLI args
    with open(f"{train_dir}/args.pkl", "wb") as f:
        pickle.dump(asdict(cfg), f, protocol=pickle.HIGHEST_PROTOCOL)

    run_with_argsd(argsd)


def load_nnet(nnet_dir: str, env: Environment) -> nn.Module:
    """Loads the neural network model.

    Args:
        nnet_dir (str): Directory of the neural network model.
        env (Environment): The environment instance.

    Returns:
        nn.Module: The loaded neural network model.
    """
    nnet_file: str = f"{nnet_dir}/model_state_dict.pt"
    if os.path.isfile(nnet_file):
        nnet = nnet_utils.load_nnet(nnet_file, env.get_env_nnet_cont())
    else:
        nnet = env.get_env_nnet_cont()

    return nnet


def load_train_state(train_dir: str, nnet_model_dir: str, env: Environment) -> tuple[nn.Module, int]:
    """Loads the training data.

    Args:
        train_dir (str): Directory of the training data.
        nnet_model_dir (str): Directory of the neural network model.
        env (Environment): The environment instance.

    Returns:
        tuple[nn.Module, int]: The neural network model and the starting iteration.
    """
    itr_file: str = f"{train_dir}/train_itr.pkl"
    if os.path.isfile(itr_file):
        with open(itr_file, "rb") as f:
            itr = pickle.load(f) + 1
    else:
        itr = 0

    nnet_model = load_nnet(nnet_model_dir, env)

    return nnet_model, itr


def step_model(
    nnet: nn.Module,
    state_episodes: list[NDArray[float32]],
    action_episodes: list[list[int]],
    start_idxs: NDArray[intp],
    device: torch.device,
    num_steps: int,
) -> tuple[Tensor, list[float], list[NDArray[float32]]]:
    """Steps the model through the given episodes.

    Args:
        nnet (nn.Module): The neural network model.
        state_episodes (list[NDArray]): List of state episodes.
        action_episodes (list[list[int]]): List of action episodes.
        start_idxs (np.NDArray): Array of start indices.
        device (torch.device): The device to run the model on.
        num_steps (int): Number of steps to predict.

    Returns:
        tuple[Tensor, list[float], list[NDArray]]: The loss, loss steps, and state episodes.
    """
    # get initial current state
    states_np: NDArray[float32] = np.stack(
        [state_episode[idx] for state_episode, idx in zip(state_episodes, start_idxs, strict=False)], axis=0
    )
    states = torch.tensor(states_np, device=device).float().contiguous()

    states_episode_arrays: list[NDArray[float32]] = [states.cpu().data.numpy()]

    loss: Tensor = torch.tensor(0.0, device=device)
    num_ex_total: int = 0
    loss_steps: list[float] = []
    for step in range(num_steps):
        # get action
        actions_np: NDArray[intp] = np.array([
            action_episode[idx] for action_episode, idx in zip(action_episodes, start_idxs + step, strict=False)
        ])
        actions = torch.tensor(actions_np, device=device).float()

        # predict next state
        states_next_pred = nnet(states, actions)

        # get ground truth next state
        states_next_gt_np: NDArray[float32] = np.stack(
            [state_episode[idx] for state_episode, idx in zip(state_episodes, start_idxs + step + 1, strict=False)],
            axis=0,
        )
        states_next_gt = torch.tensor(states_next_gt_np, device=device).float().contiguous()

        # loss_step
        loss_step = torch.pow(states_next_pred - states_next_gt, 2)
        loss_steps.append(float(loss_step.mean().cpu().data.numpy()))

        loss += loss_step.mean(dim=(1, 2, 3)).sum()
        num_ex_total += loss_step.shape[0]

        states_tens = states_next_pred.detach()

        states_episode_arrays.append(states_tens.cpu().data.numpy())

    loss /= float(num_ex_total)

    return loss, loss_steps, states_episode_arrays


def get_batch(
    state_episodes: list[NDArray[float32]],
    action_episodes: list[list[int]],
    episode_lens: NDArray[intp],
    batch_size: int,
    num_steps: int,
) -> tuple[list[NDArray[float32]], list[list[int]], NDArray[intp]]:
    """Gets a batch of episodes.

    Args:
        state_episodes (list[NDArray]): List of state episodes.
        action_episodes (list[list[int]]): List of action episodes.
        episode_lens (np.NDArray): Array of episode lengths.
        batch_size (int): Batch size.
        num_steps (int): Number of steps to predict.

    Returns:
        tuple[list[NDArray], list[list[int]], np.array]: The state episodes batch, action
            episodes batch, and start indices.
    """
    episode_idxs: NDArray[intp] = np.random.randint(len(state_episodes), size=batch_size).astype(intp)
    start_idxs: NDArray[float64] = np.random.uniform(0, 1, size=batch_size) * (
        episode_lens[episode_idxs] - num_steps - 1
    )
    start_idxs_int: NDArray[intp] = start_idxs.round().astype(intp)

    state_episodes_batch: list[NDArray[float32]] = [state_episodes[idx] for idx in episode_idxs]
    action_episodes_batch: list[list[int]] = [action_episodes[idx] for idx in episode_idxs]

    return state_episodes_batch, action_episodes_batch, start_idxs_int


def train_nnet(
    nnet: nn.Module,
    state_episodes_train: list[NDArray[float32]],
    action_episodes_train: list[list[int]],
    state_episodes_val: list[NDArray[float32]],
    action_episodes_val: list[list[int]],
    num_steps_max: int,
    device: torch.device,
    batch_size: int,
    num_itrs: int,
    start_itr: int,
    lr: float,
    lr_d: float,
    model_dir: str,
    num_steps: int = 1,
) -> None:
    """Trains the neural network model.

    Args:
        nnet (nn.Module): The neural network model.
        state_episodes_train (list[NDArray]): List of training state episodes.
        action_episodes_train (list[list[int]]): List of training action episodes.
        state_episodes_val (list[NDArray]): List of validation state episodes.
        action_episodes_val (list[list[int]]): List of validation action episodes.
        num_steps_max (int): Maximum number of steps to predict.
        device (torch.device): The device to run the model on.
        batch_size (int): Batch size.
        num_itrs (int): Number of iterations.
        start_itr (int): Starting iteration.
        lr (float): Learning rate.
        lr_d (float): Learning rate decay.
        model_dir (str): Directory to save the model.
        num_steps (int, optional): Number of steps to predict. Defaults to 1.
    """
    nnet.train()
    episode_lens_train: NDArray[intp] = np.array([state_episode.shape[0] for state_episode in state_episodes_train])
    episode_lens_val: NDArray[intp] = np.array([state_episode.shape[0] for state_episode in state_episodes_val])

    # optimization
    val_interval: int = 100
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=lr)

    # initialize status tracking
    start_time_all = time.time()

    for train_itr in range(start_itr, num_itrs):
        batch_size_eff: int = int(np.ceil(batch_size / num_steps))

        # zero the parameter gradients
        optimizer.zero_grad()
        lr_itr: float = lr * (lr_d**train_itr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_itr

        # do steps
        state_episodes_batch, action_episodes_batch, start_idxs = get_batch(
            state_episodes_train, action_episodes_train, episode_lens_train, batch_size_eff, num_steps
        )

        loss, loss_steps, _ = step_model(
            nnet, state_episodes_batch, action_episodes_batch, start_idxs, device, num_steps
        )

        # backwards
        loss.backward()

        # step
        optimizer.step()

        # display progress
        if train_itr % val_interval == 0 or train_itr == num_itrs - 1:
            print("\n-------")
            nnet.eval()

            print("Train")
            for step, loss_step in enumerate(loss_steps):
                print(f"step: {step}, loss: {loss_step:.2E}")
            print(f"Total loss: {loss.item():.2E}")

            # validation
            state_episodes_val, action_episodes_val, start_idxs_val = get_batch(
                state_episodes_val, action_episodes_val, episode_lens_val, batch_size_eff, num_steps
            )

            loss_val, loss_steps_val, _ = step_model(
                nnet, state_episodes_val, action_episodes_val, start_idxs_val, device, num_steps
            )

            print("\nValidation")
            for step, loss_step in enumerate(loss_steps_val):
                print(f"step: {step}, loss: {loss_step:.2E} (val)")

            print(
                f"Total loss: {loss_val.item():.2E} (val)\n"
                f"Itr: {train_itr}, lr: {lr_itr:.2E}, times - all: {time.time() - start_time_all:.2f}"
            )

            torch.save(nnet.state_dict(), f"{model_dir}/model_state_dict.pt")
            with open(f"{model_dir}/train_itr.pkl", "wb") as f:
                pickle.dump(train_itr, f, protocol=pickle.HIGHEST_PROTOCOL)

            if (loss_steps_val[-1] < 1e-4) and (num_steps < num_steps_max):
                print("Incrementing number of steps")
                num_steps += 1
            print("")

            nnet.train()
            start_time_all = time.time()


def main() -> None:
    """Main function to run the training process."""
    parser: ArgumentParser = ArgumentParser()
    args_dict: dict[str, Any] = parse_arguments(parser)
    run_with_argsd(args_dict)


if __name__ == "__main__":
    main()
