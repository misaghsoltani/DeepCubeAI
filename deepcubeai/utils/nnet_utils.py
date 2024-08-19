import os
import re
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


def get_device() -> Tuple[torch.device, List[int], bool]:
    """
    Gets the appropriate device for computation (CPU, CUDA, or MPS).

    Returns:
        Tuple[torch.device, List[int], bool]: A tuple containing the device, list of device IDs,
            and a boolean indicating if GPU is used.
    """
    device: torch.device = torch.device("cpu")
    devices: List[int] = []
    on_gpu: bool = False

    if ("CUDA_VISIBLE_DEVICES" in os.environ) and torch.cuda.is_available():
        device = torch.device("cuda:0")
        devices = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        on_gpu = True

    # TODO: MPS is not tested yet
    # elif torch.backends.mps.is_available():
    #     print("\n============\nWARNING: The Metal Performance Shaders (MPS) backend is being ",
    #           "used for GPU training acceleration with torch.float32. However, this code has ",
    #           "not been tested on the MPS backend!\n============\n")
    #     torch.set_default_dtype(torch.float32)
    #     device = torch.device("mps")
    #     devices = [0]
    #     on_gpu = True

    return device, devices, on_gpu


def load_nnet(model_file: str,
              nnet: nn.Module,
              device: Optional[torch.device] = None) -> nn.Module:
    """
    Loads a neural network from a file.

    Args:
        model_file (str): Path to the model file.
        nnet (nn.Module): The neural network module to load the state dict into.
        device (Optional[torch.device]): The device to map the model to.

    Returns:
        nn.Module: The loaded neural network.
    """
    if device is None:
        state_dict = torch.load(model_file, map_location=torch.device("cpu"))
    else:
        state_dict = torch.load(model_file, map_location=device)

    # remove module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = re.sub(r"^module\.", "", k)
        new_state_dict[k] = v

    # set state dict
    nnet.load_state_dict(new_state_dict)

    nnet.eval()
    nnet.to(device)

    return nnet


def get_heuristic_fn(
        nnet: nn.Module,
        device: torch.device,
        clip_zero: bool = False,
        batch_size: Optional[int] = None) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a heuristic function that computes the cost to go from states to goal states.

    Args:
        nnet (nn.Module): The neural network module.
        device (torch.device): The device to run the computation on.
        clip_zero (bool, optional): Whether to clip the cost to zero. Defaults to False.
        batch_size (Optional[int], optional): The batch size for processing. Defaults to None.

    Returns:
        Callable[[np.ndarray, np.ndarray], np.ndarray]: The heuristic function.
    """
    nnet.eval()

    def heuristic_fn(states_np: np.ndarray, states_goal_np: np.ndarray) -> np.ndarray:
        cost_to_go_l: List[np.ndarray] = []
        num_states: int = states_np.shape[0]

        batch_size_inst: int = num_states
        if batch_size is not None:
            batch_size_inst = batch_size

        start_idx: int = 0
        while start_idx < num_states:
            # get batch
            end_idx: int = min(start_idx + batch_size_inst, num_states)

            # convert batch to tensor
            states_batch = torch.tensor(states_np[start_idx:end_idx], device=device)
            states_goal_batch = torch.tensor(states_goal_np[start_idx:end_idx], device=device)

            cost_to_go_batch: np.ndarray = nnet(states_batch, states_goal_batch).cpu().data.numpy()
            cost_to_go_l.append(cost_to_go_batch)

            start_idx = end_idx

        cost_to_go = np.concatenate(cost_to_go_l, axis=0)
        assert cost_to_go.shape[0] == num_states, (f"Shape of cost_to_go is {cost_to_go.shape} "
                                                   f"num states is {num_states}")

        if clip_zero:
            cost_to_go = np.maximum(cost_to_go, 0.0)

        return cost_to_go

    return heuristic_fn


def get_model_fn(
        nnet: nn.Module,
        device: torch.device,
        batch_size: Optional[int] = None) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a model function that computes the next states given current states and actions.

    Args:
        nnet (nn.Module): The neural network module.
        device (torch.device): The device to run the computation on.
        batch_size (Optional[int], optional): The batch size for processing. Defaults to None.

    Returns:
        Callable[[np.ndarray, np.ndarray], np.ndarray]: The model function.
    """
    nnet.eval()

    def model_fn(states_np: np.ndarray, actions_np: np.ndarray) -> np.ndarray:
        states_next_l: List[np.ndarray] = []
        num_states: int = states_np.shape[0]

        batch_size_inst: int = num_states
        if batch_size is not None:
            batch_size_inst = batch_size

        start_idx: int = 0
        while start_idx < num_states:
            # get batch
            end_idx: int = min(start_idx + batch_size_inst, num_states)

            states_batch_np: np.ndarray = states_np[start_idx:end_idx]
            actions_batch_np: np.ndarray = actions_np[start_idx:end_idx]

            # get nnet output
            states_batch = torch.tensor(states_batch_np, device=device).float()
            actions_batch = torch.tensor(actions_batch_np, device=device).float()

            states_next_batch_np: np.ndarray = nnet(states_batch, actions_batch).cpu().data.numpy()
            states_next_l.append(states_next_batch_np.round().astype(np.uint8))

            start_idx = end_idx

        states_next_np = np.concatenate(states_next_l, axis=0)
        assert states_next_np.shape[0] == num_states

        return states_next_np

    return model_fn


def get_available_gpu_nums() -> List[int]:
    """
    Gets the list of available GPU numbers from the environment variable.

    Returns:
        List[int]: A list of available GPU numbers.
    """
    gpu_nums: List[int] = []
    if ("CUDA_VISIBLE_DEVICES" in os.environ) and (len(os.environ["CUDA_VISIBLE_DEVICES"]) > 0):
        gpu_nums = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]

    return gpu_nums


def load_heuristic_fn(
        nnet_dir: str,
        device: torch.device,
        on_gpu: bool,
        nnet: nn.Module,
        clip_zero: bool = False,
        gpu_num: int = -1,
        batch_size: Optional[int] = None) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Loads a heuristic function from a neural network.

    Args:
        nnet_dir (str): Directory containing the neural network model.
        device (torch.device): The device to run the computation on.
        on_gpu (bool): Whether to use GPU.
        nnet (nn.Module): The neural network module.
        clip_zero (bool, optional): Whether to clip the cost to zero. Defaults to False.
        gpu_num (int, optional): The GPU number to use. Defaults to -1.
        batch_size (Optional[int], optional): The batch size for processing. Defaults to None.

    Returns:
        Callable[[np.ndarray, np.ndarray], np.ndarray]: The heuristic function.
    """
    if (gpu_num >= 0) and on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    model_file = f"{nnet_dir}/model_state_dict.pt"

    nnet = load_nnet(model_file, nnet, device=device)
    nnet.eval()
    nnet.to(device)
    if on_gpu:
        nnet = nn.DataParallel(nnet)

    heuristic_fn = get_heuristic_fn(nnet, device, clip_zero=clip_zero, batch_size=batch_size)

    return heuristic_fn


def load_model_fn(
        model_file: str,
        device: torch.device,
        on_gpu: bool,
        nnet: nn.Module,
        gpu_num: int = -1,
        batch_size: Optional[int] = None) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Loads a model function from a neural network.

    Args:
        model_file (str): Path to the model file.
        device (torch.device): The device to run the computation on.
        on_gpu (bool): Whether to use GPU.
        nnet (nn.Module): The neural network module.
        gpu_num (int, optional): The GPU number to use. Defaults to -1.
        batch_size (Optional[int], optional): The batch size for processing. Defaults to None.

    Returns:
        Callable[[np.ndarray, np.ndarray], np.ndarray]: The model function.
    """
    if (gpu_num >= 0) and on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    nnet = load_nnet(model_file, nnet, device=device)
    nnet.eval()
    nnet.to(device)
    if on_gpu:
        nnet = nn.DataParallel(nnet)

    model_fn = get_model_fn(nnet, device, batch_size=batch_size)

    return model_fn
