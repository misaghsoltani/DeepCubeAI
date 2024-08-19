import os
import pickle
import sys
import time
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from deepcubeai.environments.environment_abstract import Environment
from deepcubeai.search_methods.gbfs_imag import gbfs, gbfs_test
from deepcubeai.utils import data_utils, env_utils, imag_utils, misc_utils, nnet_utils
from deepcubeai.utils.data_utils import print_args
from deepcubeai.utils.update_utils import q_update


def parse_arguments(parser: ArgumentParser) -> Dict[str, Any]:
    """Parses command line arguments.

    Args:
        parser (ArgumentParser): The argument parser.

    Returns:
        Dict[str, Any]: A dictionary of parsed arguments.
    """
    # Environment
    parser.add_argument("--env", type=str, required=True, help="Environment")
    parser.add_argument("--env_model", type=str, required=True, help="Environment model file")

    # Data
    parser.add_argument("--train", type=str, required=True, help="Location of training data")
    parser.add_argument("--val", type=str, required=True, help="Location of validation data")
    parser.add_argument(
        "--per_eq_tol",
        type=float,
        required=True,
        help="Percent of latent state elements that need to be equal to declare equal")

    # Debug
    parser.add_argument("--debug", action="store_true", default=False, help="")

    # Gradient Descent
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument(
        "--lr_d",
        type=float,
        default=0.9999993,
        help="Learning rate decay for every iteration. Learning rate is decayed according "
        "to: lr * (lr_d ^ itr)")

    # Training
    parser.add_argument("--max_itrs",
                        type=int,
                        default=1000000,
                        help="Maximum number of iterations")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument(
        "--single_gpu_training",
        action="store_true",
        default=False,
        help="If set, train only on one GPU. Update step will still use all GPUs given by "
        "CUDA_VISIBLE_DEVICES")
    parser.add_argument("--rb_itrs",
                        type=int,
                        default=1,
                        help="Number of iterations worth of data contained in replay buffer.")

    # Update
    parser.add_argument(
        "--loss_thresh",
        type=float,
        default=0.05,
        help="When the loss falls below this value, the target network is updated to the current"
        " network.")
    parser.add_argument(
        "--update_itrs",
        type=str,
        default="",
        help="Iterations at which to update. Last itr will be max_itrs. Update defaults to "
        "loss_thresh if empty.")
    parser.add_argument(
        "--states_per_update",
        type=int,
        default=100000,
        help="How many states to train on before checking if target network should be updated")
    parser.add_argument(
        "--update_nnet_batch_size",
        type=int,
        default=1000,
        help="Batch size of each nnet used for each process update. Make smaller if running out of"
        " memory.")
    parser.add_argument(
        "--max_solve_steps",
        type=int,
        default=1,
        help="Number of steps to take when trying to solve training states with greedy best-first "
        "search (GBFS). Each state encountered when solving is added to the training set. "
        "Number of steps starts at 1 and is increased every update until the maximum number "
        "is reached. Increasing this number can make the cost-to-go function more robust by "
        "exploring more of the state space.")

    parser.add_argument(
        "--eps_max",
        type=float,
        default=0.1,
        help=("When adding training states with GBFS, each instance will have an eps that is " +
              "distributed randomly between 0 and eps_max."))

    # Testing
    parser.add_argument("--num_test", type=int, default=1000, help="Number of test states.")

    # Data
    parser.add_argument(
        "--start_steps",
        type=int,
        required=True,
        help="Maximum number of steps to take from offline states to generate start states")
    parser.add_argument(
        "--goal_steps",
        type=int,
        required=True,
        help="Maximum number of steps to take from the start states to generate goal states")

    # Model
    parser.add_argument("--nnet_name", type=str, required=True, help="Name of neural network")
    parser.add_argument("--save_dir",
                        type=str,
                        default="saved_heur_models",
                        help="Directory to which to save model")

    # Parse arguments
    args = parser.parse_args()
    args_dict: Dict[str, Any] = vars(args)

    if len(args_dict["update_itrs"]) > 0:
        args_dict["update_itrs"] = [int(float(x)) for x in args_dict["update_itrs"].split(",")]
        args_dict["max_itrs"] = args_dict["update_itrs"][-1]
        print(f"Update iterations: {args_dict['update_itrs']}")

    # Make save directory
    model_dir: str = f"{args_dict['save_dir']}/{args_dict['nnet_name']}/"
    args_dict["model_dir"] = model_dir
    args_dict["targ_dir"] = f"{model_dir}/target/"
    args_dict["curr_dir"] = f"{model_dir}/current/"
    args_dict["output_save_loc"] = f"{model_dir}/output.txt"

    if not os.path.exists(args_dict["targ_dir"]):
        os.makedirs(args_dict["targ_dir"])

    if not os.path.exists(args_dict["curr_dir"]):
        os.makedirs(args_dict["curr_dir"])

    # Save args
    args_save_loc = f"{model_dir}/args.pkl"
    print(f"Saving arguments to {args_save_loc}")
    with open(args_save_loc, "wb") as f:
        pickle.dump(args, f, protocol=-1)

    return args_dict


def train_nnet(dqn: nn.Module,
               states_start_np: np.ndarray,
               states_goal_np: np.ndarray,
               actions_np: np.ndarray,
               ctgs_np: np.ndarray,
               batch_size: int,
               device: torch.device,
               on_gpu: bool,
               num_itrs: int,
               train_itr: int,
               lr: float,
               lr_d: float,
               display: bool = True) -> float:
    """Trains the Deep Q-Network.

    Args:
        dqn (nn.Module): The DQN model.
        states_start_np (np.ndarray): Start states as numpy array.
        states_goal_np (np.ndarray): Goal states as numpy array.
        actions_np (np.ndarray): Actions as numpy array.
        ctgs_np (np.ndarray): Cost-to-go values as numpy array.
        batch_size (int): Batch size.
        device (torch.device): Device to run the computations on.
        on_gpu (bool): Whether to use GPU.
        num_itrs (int): Number of iterations.
        train_itr (int): Current training iteration.
        lr (float): Learning rate.
        lr_d (float): Learning rate decay.
        display (bool, optional): Whether to display progress. Defaults to True.

    Returns:
        float: The last loss value.
    """
    # Initialization
    dqn.train()
    num_exs = states_start_np.shape[0]
    assert batch_size <= num_exs, ("Batch size should be less than or equal to number of train " +
                                   "examples")
    rand_batch_idxs = np.random.permutation(num_exs)
    start_batch_idx: int = 0
    end_batch_idx = start_batch_idx + batch_size

    # Optimization
    max_itrs: int = train_itr + num_itrs
    display_itrs = 100
    optimizer: Optimizer = optim.Adam(dqn.parameters(), lr=lr)

    # Status tracking
    start_time_itr = time.time()
    times: OrderedDict[str, float] = OrderedDict([("fprop", 0.0), ("bprop", 0.0), ("itr", 0.0)])

    last_loss: float = np.inf
    while train_itr < max_itrs:
        # Zero the parameter gradients
        optimizer.zero_grad()
        lr_itr: float = lr * (lr_d**train_itr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_itr

        # DQN
        start_time = time.time()
        batch_idxs = rand_batch_idxs[start_batch_idx:end_batch_idx]

        states_start: Tensor = torch.tensor(states_start_np[batch_idxs], device=device).float()
        states_goal: Tensor = torch.tensor(states_goal_np[batch_idxs], device=device).float()
        actions: Tensor = torch.tensor(actions_np[batch_idxs], device=device).unsqueeze(1)
        ctgs_targ: Tensor = torch.tensor(ctgs_np[batch_idxs], device=device).float()

        ctgs_nnet = dqn(states_start, states_goal)
        ctgs_nnet_act = ctgs_nnet.gather(1, actions)[:, 0]

        misc_utils.record_time(times, "fprop", start_time, on_gpu)

        # Backprop and step
        start_time = time.time()

        nnet_minus_targ = ctgs_nnet_act - ctgs_targ
        squared_err = torch.pow(nnet_minus_targ, 2)
        abs_err = torch.abs(nnet_minus_targ)
        huber_err = 0.5 * squared_err * (abs_err <= 1.0) + (abs_err - 0.5) * (abs_err > 1.0)

        loss = (squared_err * (nnet_minus_targ >= 0) + huber_err * (nnet_minus_targ < 0)).mean()
        loss.backward()
        optimizer.step()

        last_loss = loss.item()

        misc_utils.record_time(times, "bprop", start_time, on_gpu)

        # Display progress
        if (train_itr % display_itrs == 0) and display:
            times["itr"] = time.time() - start_time_itr
            time_str: str = misc_utils.get_time_str(times)
            print(f"Itr: {train_itr}, "
                  f"lr: {lr_itr:.2E}, "
                  f"loss: {loss.item():.2E}, "
                  f"targ_ctg: {ctgs_targ.mean().item():.2f}, "
                  f"nnet_ctg: {ctgs_nnet_act.mean().item():.2f}, "
                  f"Times - {time_str}")

            start_time_itr = time.time()
            for key in times.keys():
                times[key] = 0.0

        # Update misc
        start_batch_idx = end_batch_idx
        end_batch_idx = start_batch_idx + batch_size
        if end_batch_idx > rand_batch_idxs.shape[0]:
            rand_batch_idxs = np.random.permutation(num_exs)
            start_batch_idx = 0
            end_batch_idx = start_batch_idx + batch_size

        train_itr += 1

    return last_loss


def load_data(env: Environment, env_model: nn.Module, states_offline_np: np.ndarray,
              device: torch.device,
              argsd: Dict[str, Any]) -> Tuple[nn.Module, int, int, np.ndarray, np.ndarray, float]:
    """Loads data for training.

    Args:
        env (Environment): The environment instance.
        env_model (nn.Module): The environment model.
        states_offline_np (np.ndarray): Offline states as numpy array.
        device (torch.device): Device to run the computations on.
        argsd (Dict[str, Any]): Dictionary of arguments.

    Returns:
        Tuple[nn.Module, int, int, np.ndarray, np.ndarray, float]: Loaded data including the DQN
            model, iteration number, update number, start states, goal states, and best solved
            percentage.
    """
    nnet_file: str = f"{argsd['curr_dir']}/model_state_dict.pt"
    if os.path.isfile(nnet_file):
        nnet = nnet_utils.load_nnet(nnet_file, env.get_dqn())
        with open(f"{argsd['curr_dir']}/status.pkl", "rb") as f:
            itr, update_num, states_start_t_np, states_goal_t_np, per_solved_best = pickle.load(f)
        print(f"Loaded with itr: {itr}, "
              f"update_num: {update_num}, "
              f"per_solved_best: {per_solved_best}")
    else:
        nnet = env.get_dqn()
        itr = 0
        update_num = 0
        per_solved_best = 0.0

        samp_idxs = np.random.randint(0, states_offline_np.shape[0], size=argsd["num_test"])
        states_start_t_np = states_offline_np[samp_idxs]
        goal_steps_samp: List[int] = list(
            np.random.randint(0, argsd["goal_steps"] + 1, size=argsd["num_test"]))
        states_goal_t_np = imag_utils.random_walk(states_start_t_np, goal_steps_samp,
                                                  env.num_actions_max, env_model, device)

    return nnet, itr, update_num, states_start_t_np.copy(), states_goal_t_np.copy(
    ), per_solved_best


def main():
    """Main function to run the training process."""
    # Arguments
    parser: ArgumentParser = ArgumentParser()
    argsd: Dict[str, Any] = parse_arguments(parser)

    writer = SummaryWriter(log_dir=argsd["model_dir"])
    if not argsd["debug"]:
        sys.stdout = data_utils.Logger(argsd["output_save_loc"], "a")

    print_args(argsd)
    print(f"HOST: {os.uname()[1]}")
    # print(f"NUM DATA PROCS: {argsd['num_procs']}")
    print(f"Batch size: {argsd['batch_size']}")
    if "SLURM_JOB_ID" in os.environ:
        print(f"SLURM JOB ID: {os.environ['SLURM_JOB_ID']}")

    # Environment
    env: Environment = env_utils.get_environment(argsd["env"])
    print(f"Num actions: {env.num_actions_max}")

    # Get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print(f"device: {device}, devices: {devices}, on_gpu: {on_gpu}")

    # Load env model
    print("Loading env model")
    env_model_file: str = f"{argsd['env_model']}/env_state_dict.pt"
    env_model = nnet_utils.load_nnet(env_model_file, env.get_env_nnet())
    env_model.eval()
    env_model.to(device)
    if on_gpu:
        env_model = nn.DataParallel(env_model)

    # Load offline data
    print("Loading offline data")
    with open(argsd["val"], "rb") as f:
        episodes = pickle.load(f)
    states_offline_np: np.ndarray = np.concatenate(episodes[0], axis=0)

    # Load DQN
    print("")
    print("Getting DQN")
    dqn: nn.Module
    itr: int
    update_num: int
    dqn, itr, update_num, states_start_t_np, states_goal_t_np, per_solved_best = load_data(
        env, env_model, states_offline_np, device, argsd)

    dqn.to(device)
    if on_gpu and (not argsd["single_gpu_training"]):
        dqn = nn.DataParallel(dqn)

    # Training
    while itr < argsd["max_itrs"]:
        max_steps: int = min(update_num + 1, argsd["max_solve_steps"])
        assert max_steps >= 1, "max_solve_steps must be at least 1"

        # Generate and DQN update
        print("")
        start_time = time.time()
        if len(argsd["update_itrs"]) > 0:
            num_train_itrs: int = argsd["update_itrs"][update_num] - itr
        else:
            num_train_itrs: int = int(np.ceil(argsd["states_per_update"] / argsd["batch_size"]))
        print(f"Target train itrs: {num_train_itrs}, Max steps: {max_steps}")

        num_gen_itrs: int = int(np.ceil(num_train_itrs / max_steps))

        dqn.eval()
        # Calculate batch size multiplier to reach update batch size
        batch_size_mult: int = int(np.ceil(argsd["update_nnet_batch_size"] / argsd["batch_size"]))

        batch_size_up: int = argsd["batch_size"] * batch_size_mult
        num_gen_itrs_up: int = int(np.ceil(num_gen_itrs / batch_size_mult))

        print(f"Generating data with batch size: {batch_size_up}, iterations: {num_gen_itrs_up}")

        s_start, s_goal, acts, ctgs, times = q_update(argsd["env"], argsd["train"], batch_size_up,
                                                      num_gen_itrs_up, argsd["start_steps"],
                                                      argsd["goal_steps"], argsd["per_eq_tol"],
                                                      max_steps, argsd["env_model"],
                                                      argsd["curr_dir"], argsd["targ_dir"], device)
        print(np.unique(ctgs.astype(int)))  # TODO remove

        time_str = misc_utils.get_time_str(times)
        print(f"Times - {time_str}, Total: {time.time() - start_time:.2f}")

        # Do Q-learning
        print("")

        # Train
        print("")
        num_train_itrs: int = int(np.ceil(s_start.shape[0] / argsd["batch_size"]))
        print(f"Training model for update number {update_num} for {num_train_itrs} iterations")
        dqn.train()
        last_loss = train_nnet(dqn, s_start, s_goal, acts, ctgs, argsd["batch_size"], device,
                               on_gpu, num_train_itrs, itr, argsd["lr"], argsd["lr_d"])
        itr += num_train_itrs

        # Save nnet
        torch.save(dqn.state_dict(), f"{argsd['curr_dir']}/model_state_dict.pt")

        # Test
        start_time = time.time()
        dqn.eval()
        env_model.eval()
        max_gbfs_steps: int = min(update_num + 1, argsd["goal_steps"])
        print(f"\nTesting with {max_gbfs_steps} GBFS steps\n"
              f"Fixed test states ({states_start_t_np.shape[0]})")
        is_solved_fixed, _ = gbfs(dqn, env_model, states_start_t_np, states_goal_t_np,
                                  argsd["per_eq_tol"], max_gbfs_steps, device)
        per_solved_fixed = 100 * float(sum(is_solved_fixed)) / float(len(is_solved_fixed))
        print(f"Greedy policy solved: {per_solved_fixed}\n"
              f"Greedy policy solved (best): {per_solved_best}")
        if per_solved_fixed > per_solved_best:
            per_solved_best = per_solved_fixed
            update_nnet: bool = True
        else:
            update_nnet: bool = False

        print("Generated test states")
        gbfs_test(states_offline_np, argsd["num_test"], dqn, env_model, env.num_actions_max,
                  argsd["goal_steps"], device, max_gbfs_steps, argsd["per_eq_tol"])
        writer.add_scalar('per_solved', per_solved_fixed, itr)
        writer.flush()
        print(f"Test time: {time.time() - start_time:.2f}")

        # clear cuda memory
        torch.cuda.empty_cache()

        # update, if needed
        print(f"Last loss was {last_loss}")

        if update_nnet:
            print("Updating target network")
            data_utils.copy_files(argsd["curr_dir"], argsd["targ_dir"])
            update_num = update_num + 1

        with open(f"{argsd['curr_dir']}/status.pkl", "wb") as f:
            pickle.dump(
                (itr, update_num, states_start_t_np, states_goal_t_np, per_solved_best),
                f,
                protocol=-1,
            )

    writer.close()
    print("Done")


if __name__ == "__main__":
    main()
