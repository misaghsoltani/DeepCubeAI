import os
import pickle
import sys
import time
from argparse import ArgumentParser
from collections import OrderedDict
from shutil import rmtree
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed
from torch import Tensor, nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from deepcubeai.environments.environment_abstract import Environment
from deepcubeai.search_methods.gbfs_imag import gbfs, gbfs_test
from deepcubeai.utils import data_utils, dist_utils, env_utils, imag_utils, misc_utils, nnet_utils
from deepcubeai.utils.data_utils import print_args


def parse_arguments(parser: ArgumentParser) -> Dict[str, Any]:
    """Parse command-line arguments.

    Args:
        parser (ArgumentParser): Argument parser instance.

    Returns:
        Dict[str, Any]: Dictionary of parsed arguments.
    """
    # Environment
    parser.add_argument("--env", type=str, required=True, help="Environment")
    parser.add_argument("--env_model", type=str, required=True, help="Environment model file")

    # data
    parser.add_argument("--train", type=str, required=True, help="Location of training data")
    parser.add_argument("--val", type=str, required=True, help="Location of validation data")
    parser.add_argument("--per_eq_tol",
                        type=float,
                        required=True,
                        help="Percent of latent state elements that need to "
                        "be equal to declare equal")

    # Debug
    parser.add_argument("--debug", action="store_true", default=False, help="")

    # Gradient Descent
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lr_d",
                        type=float,
                        default=0.9999993,
                        help="Learning rate decay for every iteration. "
                        "Learning rate is decayed according to: "
                        "lr * (lr_d ^ itr)")

    # Training
    parser.add_argument("--max_itrs",
                        type=int,
                        default=1000000,
                        help="Maxmimum number of iterations")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--rb_itrs",
                        type=int,
                        default=1,
                        help="Number of iterations worth of data contained in replay "
                        "buffer.")

    # Update
    parser.add_argument("--loss_thresh",
                        type=float,
                        default=0.05,
                        help="When the loss falls below this value, "
                        "the target network is updated to the current "
                        "network.")
    parser.add_argument("--update_itrs",
                        type=str,
                        default="",
                        help="Iterations at which to udpate. "
                        "Last itr will be max_itrs. "
                        "Update defaults to loss_thresh if empty.")
    parser.add_argument("--states_per_update",
                        type=int,
                        default=100000,
                        help="How many states to train on before "
                        "checking if target network should be "
                        "updated")
    # parser.add_argument('--num_procs', type=int, default=1, help="Number of parallel workers "
    #                     "used to generate data")
    parser.add_argument("--update_nnet_batch_size",
                        type=int,
                        default=1000,
                        help="Batch size of each nnet used for "
                        "each process update. "
                        "Make smaller if running out of "
                        "memory.")
    parser.add_argument("--max_solve_steps",
                        type=int,
                        default=1,
                        help="Number of steps to take when trying to "
                        "solve training states with "
                        "greedy best-first search (GBFS). "
                        "Each state "
                        "encountered when solving is added to the "
                        "training set. Number of steps starts at "
                        "1 and is increased every update until "
                        "the maximum number is reached. "
                        "Increasing this number "
                        "can make the cost-to-go function more "
                        "robust by exploring more of the "
                        "state space.")

    parser.add_argument("--eps_max",
                        type=float,
                        default=0.1,
                        help="When adding training states with GBFS, each "
                        "instance will have an eps that is distributed "
                        "randomly between 0 and epx_max.")
    # Testing
    parser.add_argument("--num_test", type=int, default=1000, help="Number of test states.")

    # data
    parser.add_argument("--start_steps",
                        type=int,
                        required=True,
                        help="Maximum number of steps to take from "
                        "offline states to generate start states")
    parser.add_argument("--goal_steps",
                        type=int,
                        required=True,
                        help="Maximum number of steps to take from the start "
                        "states to generate goal states")

    # model
    parser.add_argument("--nnet_name", type=str, required=True, help="Name of neural network")
    parser.add_argument("--save_dir",
                        type=str,
                        default="saved_heur_models",
                        help="Director to which to save model")
    # parser.add_argument("--random_seed", type=int, default=42, help="Random seed. Used only when"
    #                     " utilizing Distributed Data Parallel")

    # parse arguments
    args = parser.parse_args()

    args_dict: Dict[str, Any] = vars(args)
    args_dict["dist_vars"] = None
    args_dict["dist_vars"] = dist_utils.get_env_vars()

    if len(args_dict["update_itrs"]) > 0:
        args_dict["update_itrs"] = [int(float(x)) for x in args_dict["update_itrs"].split(",")]
        args_dict["max_itrs"] = args_dict["update_itrs"][-1]

        print(f"Update iterations: {args_dict['update_itrs']}")

    # make save directory
    model_dir: str = f"{args_dict['save_dir']}/{args_dict['nnet_name']}/"
    args_dict["model_dir"] = model_dir
    args_dict["targ_dir"] = f"{model_dir}/target/"
    args_dict["curr_dir"] = f"{model_dir}/current/"
    args_dict["output_save_loc"] = f"{model_dir}/output.txt"
    args_dict["tmp_dir"] = f"{args_dict['curr_dir']}/tmp_her_data"

    if args_dict["dist_vars"]["world_rank"] == 0:
        if not os.path.exists(args_dict["targ_dir"]):
            os.makedirs(args_dict["targ_dir"])

        if not os.path.exists(args_dict["curr_dir"]):
            os.makedirs(args_dict["curr_dir"])

        if not os.path.exists(args_dict["tmp_dir"]):
            os.makedirs(args_dict["tmp_dir"])

        # save args
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
    """Train the Deep Q-network.

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
    # initialization
    dqn.train()
    num_exs = states_start_np.shape[0]
    assert (batch_size
            <= num_exs), "batch size should be less than or eq to number of train examples"
    rand_batch_idxs = np.random.permutation(num_exs)
    start_batch_idx: int = 0
    end_batch_idx = start_batch_idx + batch_size

    # optimization
    max_itrs: int = train_itr + num_itrs
    display_itrs = 100
    # criterion = nn.SmoothL1Loss()
    optimizer: Optimizer = optim.Adam(dqn.parameters(), lr=lr)

    # status tracking
    start_time_itr = time.time()
    times: OrderedDict[str, float] = OrderedDict([("fprop", 0.0), ("bprop", 0.0), ("itr", 0.0)])

    last_loss: float = np.inf
    while train_itr < max_itrs:
        # zero the parameter gradients
        optimizer.zero_grad()
        lr_itr: float = lr * (lr_d**train_itr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_itr

        # dqn
        start_time = time.time()
        batch_idxs = rand_batch_idxs[start_batch_idx:end_batch_idx]

        states_start: Tensor = torch.tensor(states_start_np[batch_idxs], device=device).float()
        states_goal: Tensor = torch.tensor(states_goal_np[batch_idxs], device=device).float()
        actions: Tensor = torch.tensor(actions_np[batch_idxs], device=device).unsqueeze(1)
        ctgs_targ: Tensor = torch.tensor(ctgs_np[batch_idxs], device=device).float()

        ctgs_nnet = dqn(states_start, states_goal)
        ctgs_nnet_act = ctgs_nnet.gather(1, actions)[:, 0]

        misc_utils.record_time(times, "fprop", start_time, on_gpu)

        # backprop and step
        start_time = time.time()

        # loss = criterion(ctgs_nnet_act, ctgs_targ)
        nnet_minus_targ = ctgs_nnet_act - ctgs_targ
        squared_err = torch.pow(nnet_minus_targ, 2)
        abs_err = torch.abs(nnet_minus_targ)
        huber_err = 0.5 * squared_err * (abs_err <= 1.0) + (abs_err - 0.5) * (abs_err > 1.0)

        loss = (squared_err * (nnet_minus_targ >= 0) + huber_err * (nnet_minus_targ < 0)).mean()
        loss.backward()
        optimizer.step()

        last_loss = loss.item()

        misc_utils.record_time(times, "bprop", start_time, on_gpu)

        # display progress
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

        # update misc
        start_batch_idx = end_batch_idx
        end_batch_idx = start_batch_idx + batch_size
        if end_batch_idx > rand_batch_idxs.shape[0]:
            rand_batch_idxs = np.random.permutation(num_exs)
            start_batch_idx: int = 0
            end_batch_idx = start_batch_idx + batch_size

        train_itr = train_itr + 1

    return last_loss


def load_data(
    env: Environment, env_model: nn.Module, states_offline_np: np.ndarray, device: torch.device,
    argsd: Dict[str, Any], rank: int
) -> Tuple[nn.Module, int, int, Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """Load data for training.

    Args:
        env (Environment): The environment instance.
        env_model (nn.Module): The environment model.
        states_offline_np (np.ndarray): Offline states as numpy array.
        device (torch.device): Device to run the computations on.
        argsd (Dict[str, Any]): Dictionary of arguments.
        rank (int): Rank of the current process.

    Returns:
        Tuple[nn.Module, int, int, Optional[np.ndarray], Optional[np.ndarray],
            Optional[float]]: Loaded data.
    """
    nnet_file: str = f"{argsd['curr_dir']}/model_state_dict.pt"
    if os.path.isfile(nnet_file):
        nnet = nnet_utils.load_nnet(nnet_file, env.get_dqn())
        if rank == 0:
            with open(f"{argsd['curr_dir']}/status.pkl", "rb") as file:
                itr, update_num, states_start_t_np, states_goal_t_np, per_solved_best = (
                    pickle.load(file))

            print(f"Loaded with itr: {itr}, "
                  f"update_num: {update_num}, "
                  f"per_solved_best: {per_solved_best}")

        else:
            with open(f"{argsd['curr_dir']}/status.pkl", "rb") as file:
                itr, update_num, _, _, _ = pickle.load(file)
            print(f"Loaded with itr: {itr}, update_num: {update_num}")

    else:
        nnet: nn.Module = env.get_dqn()
        itr: int = 0
        update_num: int = 0

        if rank == 0:
            per_solved_best: float = 0.0

            samp_idxs = np.random.randint(0, states_offline_np.shape[0], size=argsd["num_test"])
            states_start_t_np = states_offline_np[samp_idxs]
            goal_steps_samp: List[int] = list(
                np.random.randint(0, argsd["goal_steps"] + 1, size=argsd["num_test"]))
            states_goal_t_np = imag_utils.random_walk(states_start_t_np, goal_steps_samp,
                                                      env.num_actions_max, env_model, device)

    if rank > 0:
        return nnet, itr, update_num, None, None, None

    return (nnet, itr, update_num, states_start_t_np.copy(), states_goal_t_np.copy(),
            per_solved_best)


def main():
    """Main function to run the training process."""
    # arguments
    parser: ArgumentParser = ArgumentParser()
    argsd: Dict[str, Any] = parse_arguments(parser)

    dist_vars: Dict[str, Any] = argsd["dist_vars"]

    Logger = sys.stdout
    if dist_vars["world_rank"] == 0:
        writer = SummaryWriter(log_dir=argsd["model_dir"])
        if not argsd["debug"]:
            Logger = data_utils.Logger(argsd["output_save_loc"], "a")

    sys.stdout = dist_utils.LoggerDist(Logger, dist_vars["world_rank"], 0, dist_vars["world_size"])

    print_args(argsd)
    print(f"HOST: {os.uname()[1]}")
    # print("NUM DATA PROCS: %i" % argsd['num_procs'])
    print(f"Batch size: {argsd['batch_size']}")
    if "SLURM_JOB_ID" in os.environ:
        print(f"SLURM JOB ID: {os.environ['SLURM_JOB_ID']}")

    dist_utils.setup_ddp(dist_vars["master_addr"], dist_vars["master_port"],
                         dist_vars["world_rank"], dist_vars["world_size"])
    tmp_her_data_file: str = f"{argsd['tmp_dir']}/tmp_her_data.npz"
    # rnd_seed_manager = dist_utils.RandomSeedManager(argsd['random_seed'])
    # rnd_seed_manager.set_random_seeds()

    # environment
    env: Environment = env_utils.get_environment(argsd["env"])
    print(f"Num actions: {env.num_actions_max}")

    # get device
    on_gpu: bool
    device: torch.device
    _, devices, on_gpu = nnet_utils.get_device()
    device_id: int = dist_vars["world_rank"] % len(devices)
    if on_gpu:
        device = torch.device(device_id)

    print(f"device: {device}, devices: {devices}, on_gpu: {on_gpu}")

    # load env model
    print("Loading env model")
    env_model_file: str = f"{argsd['env_model']}/env_state_dict.pt"
    env_model = nnet_utils.load_nnet(env_model_file, env.get_env_nnet())
    env_model.eval()
    env_model.to(device)

    # load offline data
    print("Loading offline data")
    with open(argsd["val"], "rb") as file:
        episodes = pickle.load(file)
    states_offline_np: np.ndarray = np.concatenate(episodes[0], axis=0)

    # load dqn
    print("")
    print("Getting DQN")
    dqn: nn.Module
    itr: int
    update_num: int
    dqn, itr, update_num, states_start_t_np, states_goal_t_np, per_solved_best = load_data(
        env, env_model, states_offline_np, device, argsd, dist_vars["world_rank"])

    dqn.to(device)
    dqn = nn.SyncBatchNorm.convert_sync_batchnorm(dqn)
    dqn = DDP(dqn, device_ids=[device_id])  # , output_device=device_id)

    # training
    while itr < argsd["max_itrs"]:
        # torch.distributed.barrier()
        max_steps: int = min(update_num + 1, argsd["max_solve_steps"])
        assert max_steps >= 1, "max_solve_steps must be at least 1"

        # generate and dqn update
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

        # rnd_seed_manager.reset_random_seeds()

        if dist_vars["world_rank"] == 0:
            s_start, s_goal, acts, ctgs, times = dist_utils.q_update_dist(
                argsd["env"], argsd["train"], batch_size_up, num_gen_itrs_up, argsd["start_steps"],
                argsd["goal_steps"], argsd["per_eq_tol"], max_steps, argsd["env_model"],
                argsd["curr_dir"], argsd["targ_dir"], dist_vars)

            print(f"Unique CTGs: {np.unique(ctgs.astype(int))}")  # TODO remove
            time_str = misc_utils.get_time_str(times)
            print(f"Times - {time_str}, Total: {time.time() - start_time:.2f}")

            print("Saving the temp generated data to file")
            save_start_time = time.time()
            # seed_value = np.random.randint(100000, size=1)
            np.savez(tmp_her_data_file,
                     s_start=s_start,
                     s_goal=s_goal,
                     acts=acts,
                     ctgs=ctgs,
                     num_exs=s_start.shape[0])
            save_done_time = time.time() - save_start_time

        torch.distributed.barrier()

        if dist_vars["world_rank"] == 0:
            print("Loading the tmp generated data from file")
            load_start_time = time.time()

        with np.load(tmp_her_data_file, mmap_mode="r") as tmp_her_data:
            num_exs = tmp_her_data["num_exs"]
            assert (argsd["batch_size"]
                    <= num_exs), "batch size should be less than or eq to number of train examples"
            # Split the data for DDP and get the chunk
            start_idx, end_indx = dist_utils.split_data(num_exs, dist_vars["world_rank"],
                                                        dist_vars["world_size"])
            s_start = tmp_her_data["s_start"][start_idx:end_indx]
            s_goal = tmp_her_data["s_goal"][start_idx:end_indx]
            acts = tmp_her_data["acts"][start_idx:end_indx]
            ctgs = tmp_her_data["ctgs"][start_idx:end_indx]

        if dist_vars["world_rank"] == 0:
            load_done_time = time.time() - load_start_time
            print(f"Generated data is ready for training. "
                  f"Times - Save to file: {save_done_time:.2f}, "
                  f"Load from file and get the chunk: {load_done_time:.2f}, "
                  f"Total: {time.time() - save_start_time:.2f}\n")

        # torch.cuda.empty_cache()

        # do Q-learning
        # train
        num_train_itrs: int = int(np.ceil(num_exs / argsd["batch_size"]))
        print(f"Training model for update number {update_num} for {num_train_itrs} iterations")
        dqn.train()
        last_loss = train_nnet(dqn, s_start, s_goal, acts, ctgs,
                               int(np.ceil(argsd["batch_size"] / dist_vars["world_size"])), device,
                               on_gpu, num_train_itrs, itr, argsd["lr"], argsd["lr_d"])
        itr += num_train_itrs

        broadcast_data: List[int] = [None]
        torch.distributed.barrier()
        if dist_vars["world_rank"] == 0:
            # save nnet
            torch.save(dqn.state_dict(), f"{argsd['curr_dir']}/model_state_dict.pt")

            dqn_unwrapped = dqn.module
            # dqn_unwrapped.to(device)

            # test
            start_time = time.time()
            dqn_unwrapped.eval()
            env_model.eval()
            max_gbfs_steps: int = min(update_num + 1, argsd["goal_steps"])

            print(f"\nTesting with {max_gbfs_steps} GBFS steps\n"
                  f"Fixed test states ({states_start_t_np.shape[0]})")

            is_solved_fixed, _ = gbfs(dqn_unwrapped, env_model, states_start_t_np,
                                      states_goal_t_np, argsd["per_eq_tol"], max_gbfs_steps,
                                      device)
            per_solved_fixed = 100 * float(sum(is_solved_fixed)) / float(len(is_solved_fixed))

            print(f"Greedy policy solved: {per_solved_fixed}\n"
                  f"Greedy policy solved (best): {per_solved_best}")

            if per_solved_fixed > per_solved_best:
                per_solved_best = per_solved_fixed
                update_nnet: bool = True
            else:
                update_nnet: bool = False

            print("Generated test states")
            gbfs_test(states_offline_np, argsd["num_test"], dqn_unwrapped, env_model,
                      env.num_actions_max, argsd["goal_steps"], device, max_gbfs_steps,
                      argsd["per_eq_tol"])

            writer.add_scalar('per_solved', per_solved_fixed, itr)
            writer.flush()

            print(f"Test time: {(time.time() - start_time):.2f}\n"
                  f"Last loss was {last_loss}")

            # Update, if needed
            if update_nnet:
                print("Updating target network")
                data_utils.copy_files(argsd["curr_dir"], argsd["targ_dir"])
                update_num += 1

            broadcast_data = [update_num]

            with open(f"{argsd['curr_dir']}/status.pkl", "wb") as file:
                pickle.dump(
                    (itr, update_num, states_start_t_np, states_goal_t_np, per_solved_best),
                    file,
                    protocol=-1)

        torch.distributed.broadcast_object_list(broadcast_data, src=0)
        update_num = broadcast_data[0]
        torch.cuda.empty_cache()

    if dist_vars["world_rank"] == 0:
        rmtree(argsd["tmp_dir"])
        writer.close()

    print("Done")
    dist_utils.shut_down_ddp()


if __name__ == "__main__":
    main()
