import os
import random
import time
from typing import IO, Any, Dict, List, Tuple

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group, rpc
from torch.multiprocessing import Queue, get_context

from deepcubeai.utils import misc_utils, nnet_utils
from deepcubeai.utils.data_utils import Logger
from deepcubeai.utils.update_utils import q_learning_runner


class RandomSeedManager:

    def __init__(self, initial_seed: int = 0):
        """Initialize the RandomSeedManager with an initial seed.

        Args:
            initial_seed (int): The initial seed value. Defaults to 0.
        """
        self.current_seed = initial_seed

    def set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.current_seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(self.current_seed)
        random.seed(self.current_seed)
        self.current_seed += 1

    def reset_random_seeds(self) -> None:
        """Reset random seeds to a new random state."""
        torch.seed()
        np.random.seed()
        random.seed()


def split_data(data_size: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Split data indices for DDP.

    Args:
        data_size (int): Total size of the data.
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.

    Returns:
        Tuple[int, int]: Start and end indices for the current process.
    """
    chunk_size: int = -(-data_size // world_size)
    total_size: int = chunk_size * world_size

    start_indices: np.ndarray = np.arange(0, total_size, chunk_size) % data_size
    end_indices: np.ndarray = start_indices + chunk_size

    end_indices = np.where(end_indices > data_size, data_size, end_indices)
    start_indices = np.minimum(start_indices, data_size - chunk_size)

    return start_indices[rank], end_indices[rank]


class LoggerDist:

    def __init__(self,
                 orig_logger: Logger,
                 rank: int,
                 print_rank: int = -1,
                 world_size: int = None):
        """Distributed logger to handle logging in a multi-process environment.

        Args:
            orig_logger (Logger): Original logger instance.
            rank (int): Rank of the current process.
            print_rank (int): Rank to print logs. Defaults to -1 for labeled print.
            world_size (int): Total number of processes. Required if print_rank is -1.
        """
        assert (print_rank
                >= -1), "print_rank must be >= 0, or -1 to label the outputs with [rank/size]"
        assert not (print_rank == -1 and world_size
                    is None), "world_size cannot be None when print_rank is -1 (labeled print)"
        self.print_rank: int = print_rank
        self.rank: int = rank
        self.orig_logger: Logger = orig_logger
        self.world_size: int = world_size
        self.null_io: IO = open(os.devnull, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        """Write a message to the log.

        Explanation:
            If print_rank is -1, then label the prints with [rank/size] label

        Args:
            message (str): Message to log.
        """
        if self.print_rank == -1:
            if message == "\n":
                self.orig_logger.write(message)
            else:
                self.orig_logger.write(f"[{self.rank}/{self.world_size}] {message}")

        elif self.print_rank == self.rank:
            self.orig_logger.write(message)

        # else:
        #     self.null_io.write(message)

    def flush(self) -> None:
        """Flush the log."""
        if self.print_rank in (self.rank, -1):
            self.orig_logger.flush()


def get_env_vars() -> Dict[str, Any]:
    """Retrieve environment variables for DDP.

    Returns:
        Dict[str, Any]: Dictionary of environment variables.
    """
    dist_vars: Dict[str, Any] = {}

    dist_vars["local_rank"] = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", None))
    dist_vars["world_size"] = int(os.environ.get("OMPI_COMM_WORLD_SIZE", None))
    dist_vars["world_rank"] = int(os.environ.get("OMPI_COMM_WORLD_RANK", None))
    dist_vars["master_addr"] = os.environ.get("MASTER_ADDR", None)
    dist_vars["master_port"] = int(os.environ.get("MASTER_PORT", None))

    assert (dist_vars["local_rank"]
            is not None), "Environment variable OMPI_COMM_WORLD_LOCAL_RANK is not set"
    assert (dist_vars["world_size"]
            is not None), "Environment variable OMPI_COMM_WORLD_SIZE is not set"
    assert (dist_vars["world_rank"]
            is not None), "Environment variable OMPI_COMM_WORLD_RANK is not set"
    assert dist_vars["master_addr"] is not None, "Environment variable MASTER_ADDR is not set"
    assert dist_vars["master_port"] is not None, "Environment variable MASTER_PORT is not set"

    return dist_vars


def q_update_dist(
    env_name: str, data_file: str, batch_size: int, num_batches: int, start_steps: int,
    goal_steps: int, per_eq_tol: float, max_steps: int, env_model_dir: str, dqn_dir: str,
    dqn_targ_dir: str, dist_vars: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Run Q-learning updates in a distributed manner.

    Args:
        env_name (str): Name of the environment.
        data_file (str): Path to the data file.
        batch_size (int): Batch size for updates.
        num_batches (int): Number of batches.
        start_steps (int): Number of start steps.
        goal_steps (int): Number of goal steps.
        per_eq_tol (float): Percent of latent state elements that need to be equal, to declare
            equal.
        max_steps (int): Maximum number of steps.
        env_model_dir (str): Directory of the environment model.
        dqn_dir (str): Directory of the DQN model.
        dqn_targ_dir (str): Directory of the target DQN model.
        dist_vars (Dict[str, Any]): Dictionary of variables used for DDP.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
            Arrays of start states, goal states, actions, costs-to-go, and timing information.
    """
    data_queue = Queue()
    data_queue_rref = rpc.RRef(data_queue)
    time_queue = Queue()
    time_queue_rref = rpc.RRef(time_queue)

    data_runner_devices = nnet_utils.get_available_gpu_nums()

    if len(data_runner_devices) == 0:
        data_runner_devices = [None]

    num_procs: int = dist_vars["world_size"]

    num_batches_l: List[int] = misc_utils.split_evenly(num_batches, num_procs)

    worker_num: int = 0
    workers_count: int = 0
    futs_lst = []
    use_dist: bool = True
    device = None  # It will be assigned inside 'q_learning_runner()'
    print(f"Rank 0 is running {num_procs} workers for generating data")
    for data_runner_idx, num_batches_idx in enumerate(num_batches_l):
        data_runner_device = data_runner_idx % len(data_runner_devices)

        worker_num += 1
        workers_count += 1
        if worker_num > dist_vars["world_size"]:
            worker_num = 1

        fut = rpc.rpc_async(
            f"worker{worker_num}",
            q_learning_runner,
            args=(env_name, data_file, batch_size, num_batches_idx, start_steps, goal_steps,
                  per_eq_tol, max_steps, env_model_dir, dqn_dir, dqn_targ_dir, data_runner_device,
                  device, data_queue_rref, time_queue_rref, use_dist),
        )
        futs_lst.append(fut)

    # get data
    start_time = time.time()
    display_steps: List[int] = list(np.linspace(1, num_batches, 10, dtype=int))
    total_num_samples: int = batch_size * max_steps * num_batches

    states_start_np: np.array = np.zeros(0, dtype=np.uint8)
    states_goal_np: np.array = np.zeros(0, dtype=np.uint8)
    actions_np: np.array = np.zeros(total_num_samples, dtype=int)
    ctgs_np: np.array = np.zeros(total_num_samples)

    start_idx: int = 0
    batch_idx: int = 0
    while batch_idx < num_batches:
        q_res = data_queue.get()
        if q_res is None:
            batch_idx += 1
            if batch_idx in display_steps:
                print(f"{100 * batch_idx / num_batches:.2f}% ({time.time() - start_time:.2f})...")

        else:
            states_start_np_i, states_goal_np_i, actions_np_i, ctgs_np_i = q_res
            if states_start_np.shape[0] == 0:
                state_dim: int = states_start_np_i.shape[1]

                states_start_np = np.zeros((total_num_samples, state_dim), dtype=np.uint8)
                states_goal_np = np.zeros((total_num_samples, state_dim), dtype=np.uint8)

            end_idx: int = start_idx + states_start_np_i.shape[0]

            states_start_np[start_idx:end_idx] = states_start_np_i
            states_goal_np[start_idx:end_idx] = states_goal_np_i
            actions_np[start_idx:end_idx] = actions_np_i
            ctgs_np[start_idx:end_idx] = ctgs_np_i

            start_idx = end_idx

    # for fut in futs_lst:
    #     fut.wait()

    states_start_np = states_start_np[:start_idx]
    states_goal_np = states_goal_np[:start_idx]
    actions_np = actions_np[:start_idx]
    ctgs_np = ctgs_np[:start_idx]
    print(f"Generated {states_start_np.shape[0]:,} states\n")

    # get times
    times = time_queue.get()
    for _ in range(1, workers_count):
        misc_utils.add_times(times, time_queue.get())

    for key, value in times.items():
        times[key] = value / workers_count

    return states_start_np, states_goal_np, actions_np, ctgs_np, times


def _get_init_method_str(master_addr: str, master_port: int) -> str:
    """Generate initialization method string for distributed processing.

    Args:
        master_addr (str): Master address.
        master_port (int): Master port.

    Returns:
        str: Initialization method string.
    """
    return f"tcp://{master_addr}:{master_port}"


def setup_ddp(master_addr: str, master_port: int, rank: int, world_size: int) -> None:
    """Setup Distributed Data Parallel (DDP) environment.

    Args:
        master_addr (str): Master address.
        master_port (int): Master port.
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
    """
    init_process_group(backend="cpu:gloo,cuda:nccl",
                       rank=rank,
                       world_size=world_size,
                       init_method=_get_init_method_str(master_addr, master_port))

    name = f"worker{rank + 1}"
    ctx = get_context("spawn")
    proc = ctx.Process(target=_setup_rpc,
                       args=(name, master_addr, master_port + 1, rank + 1, world_size + 1, True))
    proc.daemon = True
    proc.start()

    if rank == 0:
        _setup_rpc("server", master_addr, master_port + 1, 0, world_size + 1, wait=False)


def _setup_rpc(name: str,
               master_addr: str,
               master_port: int,
               rank: int,
               world_size: int,
               wait: bool = False) -> None:
    """Setup RPC for distributed processing.

    Args:
        name (str): Name of the RPC.
        master_addr (str): Master address.
        master_port (int): Master port.
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        wait (bool): Whether to wait for shutdown. Defaults to False.
    """
    options = {
        "rpc_backend_options":
        rpc.TensorPipeRpcBackendOptions(init_method=_get_init_method_str(master_addr,
                                                                         master_port), )
    }
    # Initialize the RPC framework
    rpc.init_rpc(name, rank=rank, world_size=world_size, **options)
    if wait:
        rpc.shutdown()


def shut_down_ddp() -> None:
    """Shutdown Distributed Data Parallel (DDP) environment."""
    # rpc.shutdown()
    destroy_process_group()
