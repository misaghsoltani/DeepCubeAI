from __future__ import annotations

from argparse import ArgumentParser
from collections import OrderedDict
from multiprocessing.context import SpawnContext, SpawnProcess
import os
import pickle
import time
from typing import Any, TypeAlias

import numpy as np
from numpy import float32, intp, uint8
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
from torch.distributed.rpc.api import RRef
from torch.multiprocessing import Queue, get_context

from deepcubeai.environments.environment_abstract import Environment
from deepcubeai.utils import env_utils, imag_utils, misc_utils, nnet_utils
from deepcubeai.utils.data_utils import print_args

DataQueueType: TypeAlias = tuple[NDArray[uint8], NDArray[uint8], NDArray[intp], NDArray[float32]] | None

TimeQueueType: TypeAlias = OrderedDict[str, float]


def parse_arguments(parser: ArgumentParser) -> dict[str, Any]:
    """Parses command line arguments for the script."""
    # Environment
    parser.add_argument("--env", type=str, required=True, help="Environment")

    # Data
    parser.add_argument("--data_enc", type=str, required=True, help="Location of encoded data")

    parser.add_argument(
        "--start_steps",
        type=int,
        required=True,
        help="Maximum number of steps to take from offline states to generate start states",
    )
    parser.add_argument(
        "--goal_steps",
        type=int,
        required=True,
        help="Maximum number of steps to take from the start states to generate goal states",
    )

    parser.add_argument("--batch_size", type=int, required=True, help="Batch size with which to generate data")
    parser.add_argument("--num_batches", type=int, required=True, help="Number of batches")

    parser.add_argument("--data_start_goal", type=str, required=True, help="Location of start goal output data")

    # model
    parser.add_argument("--env_model", type=str, required=True, help="Directory of environment model")

    # parse arguments
    args = parser.parse_args()
    args_dict: dict[str, Any] = vars(args)
    print_args(args)

    return args_dict


class ZeroModel(nn.Module):
    """A zero model that returns a tensor of zeros with the shape of (batch_size, num_actions_max)."""

    def __init__(self, num_actions_max: int, device: torch.device) -> None:
        super().__init__()
        self.num_actions_max: int = num_actions_max
        self.device = device

    def forward(self, states: Tensor, _: Any) -> Tensor:
        """Forward pass that returns a tensor of zeros with the shape of (batch_size, num_actions_max)."""
        return torch.zeros((states.shape[0], self.num_actions_max), device=self.device)


# Update q-learning
def sample_boltzmann(qvals: Tensor, temp: float) -> Tensor:
    """Samples actions from Boltzmann distribution based on Q-values."""
    exp_vals: Tensor = torch.exp((1.0 / temp) * (-qvals + qvals.min(dim=1, keepdim=True)[0]))
    probs: Tensor = exp_vals / torch.sum(exp_vals, dim=1, keepdim=True)
    actions: Tensor = torch.multinomial(probs, 1)[:, 0]

    return actions


@torch.no_grad()
def q_step(
    states: Tensor,
    states_goal: Tensor,
    per_eq_tol: float,
    env_model: nn.Module,
    dqn: nn.Module,
    dqn_targ: nn.Module,
    on_gpu: bool,
    times: TimeQueueType,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Performs a single Q-learning step."""
    # get action
    start_time = time.time()
    qvals = dqn(states, states_goal).detach()
    misc_utils.record_time(times, "qvals", start_time, on_gpu)

    start_time = time.time()
    actions = sample_boltzmann(qvals, 1 / 3.0)
    misc_utils.record_time(times, "samp_acts", start_time, on_gpu)

    # check is solved
    start_time = time.time()
    is_solved = (100 * torch.mean(torch.eq(states, states_goal).float(), dim=1)) >= per_eq_tol
    misc_utils.record_time(times, "is_solved", start_time, on_gpu)

    # get next states
    start_time = time.time()
    states_next = env_model(states, actions).round().detach()
    misc_utils.record_time(times, "env_model", start_time, on_gpu)

    # min cost-to-go for next state
    start_time = time.time()
    ctg_acts_next = torch.clamp(dqn_targ(states_next, states_goal).detach(), min=0)
    ctgs_next = torch.min(ctg_acts_next, dim=1)[0]

    misc_utils.record_time(times, "ctgs", start_time, on_gpu)

    # backup cost-to-go
    start_time = time.time()
    ctg_backups = 1.0 + ctgs_next  # TODO account for varying transition costs
    ctg_backups *= 1.0 - is_solved.float()
    misc_utils.record_time(times, "backup", start_time, on_gpu)

    return states_next, actions, ctg_backups, is_solved


@torch.no_grad()
def q_learning_runner(
    env_name: str,
    data_file: str,
    batch_size: int,
    num_batches: int,
    start_steps: int,
    goal_steps: int,
    per_eq_tol: float,
    max_steps: int,
    env_model_dir: str,
    dqn_dir: str,
    dqn_targ_dir: str,
    gpu_num: int | None,
    device: torch.device,
    data_queue: Queue[DataQueueType] | RRef[Queue[DataQueueType]],
    time_queue: Queue[TimeQueueType] | RRef[Queue[TimeQueueType]],
    use_dist: bool = False,
) -> None:
    """Q-learning data generation runner for multiprocessing.

    Generates training data for Q-learning by performing random walks and computing
    Q-values, actions, and cost-to-go estimates.

    Args:
        env_name: Name of the environment
        data_file: Path to the data file containing episodes
        batch_size: Number of samples per batch
        num_batches: Total number of batches to process
        start_steps: Maximum steps for generating start states
        goal_steps: Maximum steps for generating goal states
        per_eq_tol: Tolerance for perfectness equation
        max_steps: Maximum number of steps for solving
        env_model_dir: Directory containing environment model
        dqn_dir: Directory containing DQN model
        dqn_targ_dir: Directory containing target DQN model
        gpu_num: GPU number to use (None for CPU)
        device: PyTorch device
        data_queue: Queue for sharing generated data
        time_queue: Queue for sharing timing information
        use_dist: Whether to use distributed training
    """
    times: TimeQueueType = OrderedDict([
        ("init", 0.0),
        ("gen", 0.0),
        ("qvals", 0.0),
        ("samp_acts", 0.0),
        ("is_solved", 0.0),
        ("env_model", 0.0),
        ("ctgs", 0.0),
        ("backup", 0.0),
        ("put", 0.0),
    ])

    start_time = time.time()
    env: Environment = env_utils.get_environment(env_name)
    num_actions: int = env.num_actions_max
    # Device selection logic:
    # Previous implementation rewrote CUDA_VISIBLE_DEVICES inside each spawned process using the *global* index.
    # That caused invalid device ordinal errors when a process set CUDA_VISIBLE_DEVICES to e.g. "1" and then
    # attempted to access cuda:1 in a namespace where only a single GPU (index 0) was now visible.
    # We instead keep the inherited visibility and select the absolute GPU index directly.
    if gpu_num is not None:
        device = torch.device(f"cuda:{gpu_num}")
        on_gpu = True
    else:
        on_gpu = False

    # get data
    with open(data_file, "rb") as f:
        episodes = pickle.load(f)
    states_np: NDArray[uint8] = np.concatenate(episodes[0], axis=0)

    # load env model nnet
    env_model_file: str = f"{env_model_dir}/env_state_dict.pt"
    env_model: nn.Module = nnet_utils.load_nnet(env_model_file, env.get_env_nnet())
    env_model.to(device)
    env_model.eval()

    # load target dqn
    dqn_targ_file: str = f"{dqn_targ_dir}/model_state_dict.pt"
    dqn_targ: nn.Module
    if not os.path.isfile(dqn_targ_file):
        dqn_targ = ZeroModel(env.num_actions_max, device)
    else:
        dqn_targ = nnet_utils.load_nnet(dqn_targ_file, env.get_dqn())

    dqn_targ.to(device)
    dqn_targ.eval()

    # load dqn
    dqn_file: str = f"{dqn_dir}/model_state_dict.pt"
    dqn: nn.Module
    if not os.path.isfile(dqn_file):
        dqn = ZeroModel(env.num_actions_max, device)
    else:
        dqn = nnet_utils.load_nnet(dqn_file, env.get_dqn())

    dqn.to(device)
    dqn.eval()

    misc_utils.record_time(times, "init", start_time, on_gpu)

    # get data
    for _ in range(num_batches):
        # get start and end states
        start_time = time.time()
        samp_idxs = np.random.randint(0, states_np.shape[0], size=batch_size)

        start_steps_samp = [start_steps] * batch_size
        goal_steps_samp: list[int] = list(np.random.randint(0, goal_steps + 1, size=batch_size))

        states_start_np = imag_utils.random_walk(
            states_np[samp_idxs].astype(float32), start_steps_samp, num_actions, env_model, device
        )  # TODO fix by sampling
        states_goal_np = imag_utils.random_walk(states_start_np, goal_steps_samp, num_actions, env_model, device)

        states_start = torch.tensor(states_start_np.astype(uint8), device=device).float()
        states_goal = torch.tensor(states_goal_np.astype(uint8), device=device).float()

        misc_utils.record_time(times, "gen", start_time, on_gpu)

        # do q-learning update
        for _step in range(max_steps):
            states_start_next, actions, ctgs, is_solved = q_step(
                states_start, states_goal, per_eq_tol, env_model, dqn, dqn_targ, on_gpu, times
            )

            start_time = time.time()

            states_start_np_u8: NDArray[uint8] = states_start.cpu().detach().numpy().astype(uint8, copy=False)
            states_goal_np_u8: NDArray[uint8] = states_goal.cpu().detach().numpy().astype(uint8, copy=False)
            actions_np: NDArray[intp] = actions.cpu().detach().numpy().astype(intp, copy=False)
            ctgs_np: NDArray[float32] = ctgs.cpu().detach().numpy().astype(float32, copy=False)

            data_tuple: DataQueueType = (states_start_np_u8, states_goal_np_u8, actions_np, ctgs_np)
            if use_dist:
                # When using distributed RPC, data_queue is an RRef - invoke put remotely
                data_queue.rpc_async().put(data_tuple).wait()  # type: ignore[union-attr]
            else:
                # Local multiprocessing queue
                data_queue.put(data_tuple)  # type: ignore[union-attr]

            misc_utils.record_time(times, "put", start_time, on_gpu)

            not_solved = torch.logical_not(is_solved)
            states_start = states_start_next[not_solved]
            states_goal = states_goal[not_solved]
            # Break if there are no remaining samples in this batch
            if states_start.shape[0] == 0:
                break

        # Signal the end of the batch
        if use_dist:
            data_queue.rpc_async().put(None).wait()  # type: ignore[union-attr]
        else:
            data_queue.put(None)  # type: ignore[union-attr]

    if use_dist:
        time_queue.rpc_async().put(times).wait()  # type: ignore[union-attr]
    else:
        time_queue.put(times)  # type: ignore[union-attr]


@torch.no_grad()
def q_update(
    env_name: str,
    data_file: str,
    batch_size: int,
    num_batches: int,
    start_steps: int,
    goal_steps: int,
    per_eq_tol: float,
    max_steps: int,
    env_model_dir: str,
    dqn_dir: str,
    dqn_targ_dir: str,
    device: torch.device,
    verbose: bool = True,
) -> tuple[NDArray[uint8], NDArray[uint8], NDArray[intp], NDArray[float32], TimeQueueType]:
    """Performs Q-learning updates using multiprocessing."""
    # get devices
    data_runner_devices_raw: list[int] = nnet_utils.get_available_gpu_nums()
    data_runner_devices: list[int | None] = list(data_runner_devices_raw)

    if len(data_runner_devices) == 0:
        data_runner_devices = [None]

    num_procs: int = len(data_runner_devices)

    # start runners
    num_batches_l: list[int] = misc_utils.split_evenly(num_batches, num_procs)

    ctx: SpawnContext = get_context("spawn")
    procs: list[SpawnProcess] = []
    queue: Queue[DataQueueType] = ctx.Queue()
    time_queue: Queue[TimeQueueType] = ctx.Queue()

    for data_runner_idx, num_batches_idx in enumerate(num_batches_l):
        data_runner_device = data_runner_devices[data_runner_idx % len(data_runner_devices)]

        proc: SpawnProcess = ctx.Process(
            target=q_learning_runner,
            args=(
                env_name,
                data_file,
                batch_size,
                num_batches_idx,
                start_steps,
                goal_steps,
                per_eq_tol,
                max_steps,
                env_model_dir,
                dqn_dir,
                dqn_targ_dir,
                data_runner_device,
                device,
                queue,
                time_queue,
            ),
        )
        proc.daemon = True
        proc.start()

        procs.append(proc)

    # get data
    start_time = time.time()
    display_steps: list[int] = list(np.linspace(1, num_batches, 10, dtype=int))
    total_num_samples: int = batch_size * max_steps * num_batches

    states_start_np: NDArray[uint8] = np.zeros(0, dtype=uint8)
    states_goal_np: NDArray[uint8] = np.zeros(0, dtype=uint8)
    actions_np: NDArray[intp] = np.zeros(total_num_samples, dtype=intp)
    ctgs_np: NDArray[float32] = np.zeros(total_num_samples, dtype=float32)

    start_idx: int = 0
    batch_idx: int = 0
    while batch_idx < num_batches:
        q_res: DataQueueType = queue.get()
        if q_res is None:
            batch_idx += 1
            if verbose and (batch_idx in display_steps):
                print(f"{100 * batch_idx / num_batches:.2f}% ({time.time() - start_time:.2f})...")

        else:
            states_start_np_i, states_goal_np_i, actions_np_i, ctgs_np_i = q_res
            if states_start_np.shape[0] == 0:
                state_dim: int = states_start_np_i.shape[1]

                states_start_np = np.zeros((total_num_samples, state_dim), dtype=uint8)
                states_goal_np = np.zeros((total_num_samples, state_dim), dtype=uint8)

            end_idx: int = start_idx + states_start_np_i.shape[0]

            states_start_np[start_idx:end_idx] = states_start_np_i
            states_goal_np[start_idx:end_idx] = states_goal_np_i
            actions_np[start_idx:end_idx] = actions_np_i
            ctgs_np[start_idx:end_idx] = ctgs_np_i

            start_idx = end_idx

    states_start_np = states_start_np[:start_idx]
    states_goal_np = states_goal_np[:start_idx]
    actions_np = actions_np[:start_idx]
    ctgs_np = ctgs_np[:start_idx]
    if verbose:
        print(f"Generated {states_start_np.shape[0]:,} states\n")

    # get times
    times = time_queue.get()
    for _ in range(1, len(procs)):
        misc_utils.add_times(times, time_queue.get())

    for key, value in times.items():
        times[key] = value / len(procs)

    # join processes
    for proc in procs:
        proc.join()

    return states_start_np, states_goal_np, actions_np, ctgs_np, times
