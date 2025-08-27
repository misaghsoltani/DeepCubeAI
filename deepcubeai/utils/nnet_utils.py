from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
import os
from types import FunctionType
from typing import TypeAlias

import numpy as np
from numpy import float32
from numpy.typing import NDArray
import torch
from torch import nn

CallableModelType: TypeAlias = Callable[[NDArray[float32], NDArray[float32]], NDArray[float32]] | FunctionType


def get_device() -> tuple[torch.device, list[int], bool]:
    """Return the available compute device, logical device IDs, and a boolean for GPU/accelerator use.

    Order of preference:
      1) CUDA/ROCm (device type 'cuda')
      2) Intel XPU (device type 'xpu')
      3) Apple Metal (device type 'mps')
      4) CPU

    Returns:
        (device, devices, on_gpu):
            device: torch.device to place new tensors/models on.
            devices: logical device indices for the selected backend (e.g., [0,1,2,...]).
            on_gpu: True if using any GPU/accelerator backend (CUDA/XPU/MPS), else False.
    """
    device: torch.device = torch.device("cpu")
    devices: list[int] = []
    on_gpu: bool = False

    # CUDA / ROCm (device type is still "cuda")
    if torch.cuda.is_available():
        num_visible = torch.cuda.device_count()
        if num_visible > 0:
            # Respect torchrun / DDP conventions
            local_rank_str = os.environ.get("LOCAL_RANK")
            try:
                local_rank = int(local_rank_str) if local_rank_str is not None else 0
            except ValueError:
                local_rank = 0
            if not (0 <= local_rank < num_visible):
                local_rank = 0

            # Pin current process to its device
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")

            devices = list(range(num_visible))  # CUDA_VISIBLE_DEVICES may remap them
            on_gpu = True
            return device, devices, on_gpu

    # Intel XPU
    # Use guard for older versions of PyTorch
    if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
        num_visible = torch.xpu.device_count()
        if num_visible > 0:
            local_rank_str = os.environ.get("LOCAL_RANK")
            try:
                local_rank = int(local_rank_str) if local_rank_str is not None else 0
            except ValueError:
                local_rank = 0
            if not (0 <= local_rank < num_visible):
                local_rank = 0

            torch.xpu.set_device(local_rank)
            device = torch.device(f"xpu:{local_rank}")
            devices = list(range(num_visible))
            on_gpu = True
            return device, devices, on_gpu

    # Apple Metal (MPS)
    # Single logical device today - no device_count API
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        devices = [0]
        on_gpu = True
        return device, devices, on_gpu

    # CPU fallback
    return device, devices, on_gpu


def load_nnet(model_file: str, nnet: nn.Module, device: torch.device | None = None) -> nn.Module:
    """Loads a neural network from a file.

    Args:
        model_file (str): Path to the model file.
        nnet (nn.Module): The neural network module to load the state dict into.
        device (torch.device | None): The device to map the model to.

    Returns:
        nn.Module: The loaded neural network.
    """
    if device is None:
        device = torch.device("cpu")

    state_dict = torch.load(model_file, map_location=device)

    # Remove common Distributed/DataParallel 'module.' prefix if present
    new_state_dict = OrderedDict({(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()})

    # set state dict
    nnet.load_state_dict(new_state_dict)

    nnet.eval()
    nnet.to(device)

    return nnet


def get_heuristic_fn(
    nnet: nn.Module, device: torch.device, clip_zero: bool = False, batch_size: int | None = None
) -> CallableModelType:
    """Returns a heuristic function that computes the cost to go from states to goal states.

    Args:
        nnet (nn.Module): The neural network module.
        device (torch.device): The device to run the computation on.
        clip_zero (bool, optional): Whether to clip the cost to zero. Defaults to False.
        batch_size (Optional[int], optional): The batch size for processing. Defaults to None.

    Returns:
        Callable[[NDArray, NDArray], NDArray]: The heuristic function.
    """
    nnet.eval()

    def heuristic_fn(states_np: NDArray[float32], states_goal_np: NDArray[float32]) -> NDArray[float32]:
        cost_to_go_l: list[NDArray[float32]] = []
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

            cost_to_go_batch: NDArray[float32] = nnet(states_batch, states_goal_batch).cpu().data.numpy()
            cost_to_go_l.append(cost_to_go_batch)

            start_idx = end_idx

        cost_to_go = np.concatenate(cost_to_go_l, axis=0)
        assert cost_to_go.shape[0] == num_states, (
            f"Shape of cost_to_go is {cost_to_go.shape} num states is {num_states}"
        )

        if clip_zero:
            cost_to_go = np.maximum(cost_to_go, 0.0)

        return cost_to_go

    return heuristic_fn


def get_model_fn(nnet: nn.Module, device: torch.device, batch_size: int | None = None) -> CallableModelType:
    """Returns a model function that computes the next states given current states and actions.

    Args:
        nnet (nn.Module): The neural network module.
        device (torch.device): The device to run the computation on.
        batch_size (Optional[int], optional): The batch size for processing. Defaults to None.

    Returns:
        Callable[[NDArray, NDArray], NDArray]: The model function.
    """
    nnet.eval()

    def model_fn(states_np: NDArray[float32], actions_np: NDArray[float32]) -> NDArray[float32]:
        states_next_l: list[NDArray[float32]] = []
        num_states: int = states_np.shape[0]

        batch_size_inst: int = num_states
        if batch_size is not None:
            batch_size_inst = batch_size

        start_idx: int = 0
        while start_idx < num_states:
            # get batch
            end_idx: int = min(start_idx + batch_size_inst, num_states)

            states_batch_np: NDArray[float32] = states_np[start_idx:end_idx]
            actions_batch_np: NDArray[float32] = actions_np[start_idx:end_idx]

            # get nnet output
            states_batch = torch.tensor(states_batch_np, device=device).float()
            actions_batch = torch.tensor(actions_batch_np, device=device).float()

            states_next_batch_np: NDArray[float32] = nnet(states_batch, actions_batch).cpu().data.numpy()
            states_next_l.append(states_next_batch_np.round().astype(float32))

            start_idx = end_idx

        states_next_np = np.concatenate(states_next_l, axis=0)
        assert states_next_np.shape[0] == num_states

        return states_next_np

    return model_fn


def get_available_gpu_nums() -> list[int]:
    """Gets the list of available GPU numbers from the environment variable.

    Returns:
        list[int]: A list of available GPU numbers.
    """
    gpu_nums: list[int] = []
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
    batch_size: int | None = None,
) -> CallableModelType:
    """Loads a heuristic function from a neural network.

    Args:
        nnet_dir (str): Directory containing the neural network model.
        device (torch.device): The device to run the computation on.
        on_gpu (bool): Whether to use GPU.
        nnet (nn.Module): The neural network module.
        clip_zero (bool, optional): Whether to clip the cost to zero. Defaults to False.
        gpu_num (int, optional): The GPU number to use. Defaults to -1.
        batch_size (Optional[int], optional): The batch size for processing. Defaults to None.

    Returns:
        Callable[[NDArray, NDArray], NDArray]: The heuristic function.
    """
    if (gpu_num >= 0) and on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    model_file = f"{nnet_dir}/model_state_dict.pt"

    nnet = load_nnet(model_file, nnet, device=device)
    nnet.eval()
    nnet.to(device)
    if on_gpu:
        nnet = nn.DataParallel(nnet)

    heuristic_fn: CallableModelType = get_heuristic_fn(nnet, device, clip_zero=clip_zero, batch_size=batch_size)

    return heuristic_fn


def load_model_fn(
    model_file: str,
    device: torch.device,
    on_gpu: bool,
    nnet: nn.Module,
    gpu_num: int = -1,
    batch_size: int | None = None,
) -> CallableModelType:
    """Loads a model function from a neural network.

    Args:
        model_file (str): Path to the model file.
        device (torch.device): The device to run the computation on.
        on_gpu (bool): Whether to use GPU.
        nnet (nn.Module): The neural network module.
        gpu_num (int, optional): The GPU number to use. Defaults to -1.
        batch_size (Optional[int], optional): The batch size for processing. Defaults to None.

    Returns:
        Callable[[NDArray, NDArray], NDArray]: The model function.
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
