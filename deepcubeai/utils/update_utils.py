import os
import pickle
import time
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.multiprocessing import get_context

from deepcubeai.environments.environment_abstract import Environment
from deepcubeai.utils import env_utils, imag_utils, misc_utils, nnet_utils
from deepcubeai.utils.data_utils import print_args


def parse_arguments(parser: ArgumentParser) -> Dict[str, Any]:
    # Environment
    parser.add_argument('--env', type=str, required=True, help="Environment")

    # Data
    parser.add_argument('--data_enc', type=str, required=True, help="Location of encoded data")

    parser.add_argument('--start_steps',
                        type=int,
                        required=True,
                        help="Maximum number of steps to take from "
                        "offline states to generate start states")
    parser.add_argument('--goal_steps',
                        type=int,
                        required=True,
                        help="Maximum number of steps to take from the start "
                        "states to generate goal states")

    parser.add_argument('--batch_size',
                        type=int,
                        required=True,
                        help="Batch size with which to generate data")
    parser.add_argument('--num_batches', type=int, required=True, help="Number of batches")

    parser.add_argument('--data_start_goal',
                        type=str,
                        required=True,
                        help="Location of start goal output data")

    # model
    parser.add_argument('--env_model',
                        type=str,
                        required=True,
                        help="Directory of environment model")

    # parse arguments
    args = parser.parse_args()
    args_dict: Dict[str, Any] = vars(args)
    print_args(args)

    return args_dict


class ZeroModel(nn.Module):

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, num_actions_max: int, device):
        super().__init__()
        self.num_actions_max: int = num_actions_max
        self.device = device

    def forward(self, states, _):
        return torch.zeros((states.shape[0], self.num_actions_max), device=self.device)


# Update q-learning
def sample_boltzmann(qvals: Tensor, temp: float) -> np.array:
    exp_vals = torch.exp((1.0 / temp) * (-qvals + qvals.min(dim=1, keepdim=True)[0]))
    probs = exp_vals / torch.sum(exp_vals, dim=1, keepdim=True)
    actions = torch.multinomial(probs, 1)[:, 0]

    return actions


def q_step(states: Tensor, states_goal: Tensor, per_eq_tol: float, env_model: nn.Module,
           dqn: nn.Module, dqn_targ: nn.Module, on_gpu: bool, times):
    # get action
    start_time = time.time()
    qvals = dqn(states, states_goal).detach()
    misc_utils.record_time(times, 'qvals', start_time, on_gpu)

    start_time = time.time()
    actions = sample_boltzmann(qvals, 1 / 3.0)
    misc_utils.record_time(times, 'samp_acts', start_time, on_gpu)

    # check is solved
    start_time = time.time()
    is_solved = (100 * torch.mean(torch.eq(states, states_goal).float(), dim=1)) >= per_eq_tol
    misc_utils.record_time(times, 'is_solved', start_time, on_gpu)

    # get next states
    start_time = time.time()
    states_next = env_model(states, actions).round().detach()
    misc_utils.record_time(times, 'env_model', start_time, on_gpu)

    # min cost-to-go for next state
    start_time = time.time()
    ctg_acts_next = torch.clamp(dqn_targ(states_next, states_goal).detach(), min=0)
    ctgs_next = torch.min(ctg_acts_next, dim=1)[0]

    misc_utils.record_time(times, 'ctgs', start_time, on_gpu)

    # backup cost-to-go
    start_time = time.time()
    ctg_backups = 1.0 + ctgs_next  # TODO account for varying transition costs
    ctg_backups = ctg_backups * (1.0 - is_solved.float())
    misc_utils.record_time(times, 'backup', start_time, on_gpu)

    return states_next, actions, ctg_backups, is_solved


def q_learning_runner(env_name: str,
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
                      gpu_num: Optional[int],
                      device,
                      data_queue,
                      time_queue,
                      use_dist: bool = False):
    times: OrderedDict[str, float] = OrderedDict([('init', 0.0), ('gen', 0.0), ('qvals', 0.0),
                                                  ('samp_acts', 0.0), ('is_solved', 0.0),
                                                  ('env_model', 0.0), ('ctgs', 0.0),
                                                  ('backup', 0.0), ('put', 0.0)])

    start_time = time.time()
    env: Environment = env_utils.get_environment(env_name)
    num_actions: int = env.num_actions_max
    if gpu_num is not None:
        if not use_dist:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
        else:
            device = torch.device(f"cuda:{gpu_num}")
        on_gpu: bool = True
    else:
        on_gpu: bool = False

    # get data
    with open(data_file, "rb") as f:
        episodes = pickle.load(f)
    states_np: np.ndarray = np.concatenate(episodes[0], axis=0)

    # load env model nnet
    env_model_file: str = f"{env_model_dir}/env_state_dict.pt"
    env_model = nnet_utils.load_nnet(env_model_file, env.get_env_nnet())
    env_model.to(device)
    env_model.eval()

    # load target dqn
    dqn_targ_file: str = f"{dqn_targ_dir}/model_state_dict.pt"
    if not os.path.isfile(dqn_targ_file):
        dqn_targ = ZeroModel(env.num_actions_max, device)
    else:
        dqn_targ = nnet_utils.load_nnet(dqn_targ_file, env.get_dqn())

    dqn_targ.to(device)
    dqn_targ.eval()

    # load dqn
    dqn_file: str = f"{dqn_dir}/model_state_dict.pt"
    if not os.path.isfile(dqn_file):
        dqn = ZeroModel(env.num_actions_max, device)
    else:
        dqn = nnet_utils.load_nnet(dqn_file, env.get_dqn())

    dqn.to(device)
    dqn.eval()

    misc_utils.record_time(times, 'init', start_time, on_gpu)

    # get data3
    for batch_idx in range(num_batches):
        # get start and end states
        start_time = time.time()
        samp_idxs = np.random.randint(0, states_np.shape[0], size=batch_size)

        start_steps_samp = [start_steps] * batch_size
        goal_steps_samp: List[int] = list(np.random.randint(0, goal_steps + 1, size=batch_size))

        states_start_np = imag_utils.random_walk(states_np[samp_idxs], start_steps_samp,
                                                 num_actions, env_model,
                                                 device)  # TODO fix by sampling
        states_goal_np = imag_utils.random_walk(states_start_np, goal_steps_samp, num_actions,
                                                env_model, device)

        states_start = torch.tensor(states_start_np.astype(np.uint8), device=device).float()
        states_goal = torch.tensor(states_goal_np.astype(np.uint8), device=device).float()

        misc_utils.record_time(times, 'gen', start_time, on_gpu)

        # do q-learning update
        for step in range(max_steps):
            states_start_next, actions, ctgs, is_solved = q_step(states_start, states_goal,
                                                                 per_eq_tol, env_model, dqn,
                                                                 dqn_targ, on_gpu, times)

            start_time = time.time()

            states_start_np = states_start.cpu().data.numpy()
            states_goal_np = states_goal.cpu().data.numpy()
            actions_np = actions.cpu().data.numpy()
            ctgs_np = ctgs.cpu().data.numpy()
            if use_dist:
                data_queue.rpc_async().put(
                    (states_start_np, states_goal_np, actions_np, ctgs_np)).wait()
            else:
                data_queue.put((states_start_np, states_goal_np, actions_np, ctgs_np))

            misc_utils.record_time(times, 'put', start_time, on_gpu)

            not_solved = torch.logical_not(is_solved)
            states_start = states_start_next[not_solved]
            states_goal = states_goal[not_solved]
            if states_start_np.shape[0] == 0:
                break

        # Signal the end of the batch
        if use_dist:
            data_queue.rpc_async().put(None).wait()
        else:
            data_queue.put(None)

    if use_dist:
        time_queue.rpc_async().put(times).wait()
    else:
        time_queue.put(times)


def q_update(env_name: str, data_file: str, batch_size: int, num_batches: int, start_steps: int,
             goal_steps: int, per_eq_tol: float, max_steps: int, env_model_dir: str, dqn_dir: str,
             dqn_targ_dir: str, device):

    # get devices
    data_runner_devices: List[Optional[int]] = nnet_utils.get_available_gpu_nums()

    if len(data_runner_devices) == 0:
        data_runner_devices = [None]

    num_procs: int = len(data_runner_devices)

    # start runners
    num_batches_l: List[int] = misc_utils.split_evenly(num_batches, num_procs)

    ctx = get_context("spawn")
    procs: List[ctx.Process] = []
    queue: ctx.Queue = ctx.Queue()
    time_queue: ctx.Queue = ctx.Queue()

    for data_runner_idx, num_batches_idx in enumerate(num_batches_l):
        data_runner_device = data_runner_devices[data_runner_idx % len(data_runner_devices)]

        proc = ctx.Process(target=q_learning_runner,
                           args=(env_name, data_file, batch_size, num_batches_idx, start_steps,
                                 goal_steps, per_eq_tol, max_steps, env_model_dir, dqn_dir,
                                 dqn_targ_dir, data_runner_device, device, queue, time_queue))
        proc.daemon = True
        proc.start()

        procs.append(proc)

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

        q_res = queue.get()
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

    states_start_np = states_start_np[:start_idx]
    states_goal_np = states_goal_np[:start_idx]
    actions_np = actions_np[:start_idx]
    ctgs_np = ctgs_np[:start_idx]
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
