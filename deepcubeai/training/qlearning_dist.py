"""Distributed Q-learning (GPU-only) FSDP2 pipeline."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import OrderedDict
import contextlib
from dataclasses import asdict, dataclass
import datetime
import os
import pickle
import random
import sys
import time
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn, optim
import torch.distributed as dist
from torch.distributed.checkpoint import state_dict_saver
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from deepcubeai.environments.environment_abstract import Environment
from deepcubeai.search_methods.gbfs_imag import gbfs, gbfs_test
from deepcubeai.utils import data_utils, dist_utils, env_utils, imag_utils, nnet_utils, update_utils_dist
from deepcubeai.utils.data_utils import print_args
from deepcubeai.utils.update_utils_dist import ReplayBuffer


class OnBatchHandler:
    """Callable training-batch consumer."""

    def __init__(
        self,
        *,
        dqn: nn.Module,
        opt: Optimizer,
        device: torch.device,
        world_size: int,
        lr: float,
        lr_d: float,
        max_itrs: int,
        micro_local_batch: int,
        micro_accum: int,
        use_autocast: bool,
        is_rank0: bool,
        display_itrs: int,
        times: OrderedDict[str, float],
        replay_max: int,
        replay_frac: float,
    ) -> None:
        self.dqn = dqn
        self.opt = opt
        self.device = device
        self.world_size = world_size
        self.lr = lr
        self.lr_d = lr_d
        self.max_itrs = max_itrs
        self.micro_local_batch = micro_local_batch
        self.micro_accum = micro_accum
        self.use_autocast = use_autocast
        self.is_rank0 = is_rank0
        self.display_itrs = display_itrs
        self.times = times

        self.itr: int = 0
        self.micro_step: int = 0
        self.last_loss_val: float = float("inf")
        self.start_time_itr: float = time.time()

        self.replay: ReplayBuffer = ReplayBuffer(replay_max, replay_frac)

    def __call__(self, states_start: Tensor, states_goal: Tensor, acts: Tensor, ctgs: Tensor) -> None:
        """Training micro-step consumer used by the async generator."""
        # Early exit: once logical iterations reach the cap, do nothing
        # This prevents overshooting when generation keeps feeding batches
        if self.itr >= self.max_itrs:
            return

        self.dqn.train()
        batch_size = states_start.shape[0]

        # Optional replay augmentation
        if self.replay.capacity > 0 and self.replay.frac > 0.0:
            replay_take = int(self.replay.frac * batch_size)
            sample = self.replay.sample(replay_take)
            if sample is not None:
                rs_s, rs_g, rs_a, rs_c = sample
                states_start = torch.cat([states_start, rs_s], dim=0)
                states_goal = torch.cat([states_goal, rs_g], dim=0)
                acts = torch.cat([acts, rs_a], dim=0)
                ctgs = torch.cat([ctgs, rs_c], dim=0)
                batch_size = states_start.shape[0]

        # Shuffle
        if batch_size > 1:
            perm_all = torch.randperm(batch_size, device=states_start.device)
            states_start = states_start[perm_all]
            states_goal = states_goal[perm_all]
            acts = acts[perm_all]
            ctgs = ctgs[perm_all]

        # Cross-rank trim to smallest available per-rank batch
        if dist.is_initialized() and self.world_size > 1:
            try:
                n_local = torch.tensor([batch_size], device=self.device, dtype=torch.int64)
                gather_list = [torch.zeros_like(n_local) for _ in range(self.world_size)]
                dist.all_gather(gather_list, n_local)
                min_n = int(torch.min(torch.stack(gather_list)).item())
            except Exception:
                min_n = batch_size

            if min_n == 0:
                return

            if batch_size != min_n:
                states_start = states_start[:min_n]
                states_goal = states_goal[:min_n]
                acts = acts[:min_n]
                ctgs = ctgs[:min_n]
                batch_size = min_n

        # Iterate over micro-batches (each ~micro_local_batch per rank)
        for start in range(0, batch_size, self.micro_local_batch):
            end = min(start + self.micro_local_batch, batch_size)
            mb_s_start = states_start[start:end].float()
            mb_s_goal = states_goal[start:end].float()
            mb_acts = acts[start:end]
            mb_ctg = ctgs[start:end].float()
            lr_now = self.lr * (self.lr_d**self.itr)
            for g in self.opt.param_groups:
                g["lr"] = lr_now

            if self.micro_accum > 1 and hasattr(self.dqn, "set_requires_gradient_sync"):
                fn = cast(Any, self.dqn).set_requires_gradient_sync
                if callable(fn):
                    fn(self.micro_step % self.micro_accum == self.micro_accum - 1)

            start_f = time.time()
            # Forward + loss under autocast (bf16) while parameters remain fp32
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_autocast):
                q_vals = self.dqn(mb_s_start, mb_s_goal)
                loss, q_act = update_utils_dist.compute_loss(q_vals, mb_acts, mb_ctg)

            self.times["fprop"] += time.time() - start_f
            loss.backward()
            self.last_loss_val = float(loss.detach())
            if self.micro_step % self.micro_accum == self.micro_accum - 1:
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                self.itr += 1
                # Compute global (mean) loss across ranks for logging
                if dist.is_initialized():
                    loss_t = torch.tensor([self.last_loss_val], dtype=torch.float32, device=self.device)
                    dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
                    last_loss_global = float(loss_t.item())
                else:
                    last_loss_global = self.last_loss_val

                if self.is_rank0 and self.itr % self.display_itrs == 0:
                    self.times["itr"] = time.time() - self.start_time_itr
                    time_str = ", ".join([f"{k}: {v:.2f}" for k, v in self.times.items()])
                    print(
                        f"Itr: {self.itr}, lr: {lr_now:.2E}, loss: {last_loss_global:.2E}, "
                        f"targ_ctg: {mb_ctg.mean().item():.2f}, "
                        f"nnet_ctg: {q_act.mean().item():.2f}, Times - {time_str}"
                    )
                    self.start_time_itr = time.time()
                    for k in self.times:
                        self.times[k] = 0.0

            self.micro_step += 1

            if self.itr >= self.max_itrs:
                break

        # Add to replay after using this batch
        self.replay.add(states_start, states_goal, acts, ctgs)


def parse_arguments(parser: ArgumentParser) -> dict[str, Any]:
    """Parse CLI args."""
    parser.add_argument("--env", type=str, required=True, help="Environment")
    parser.add_argument("--env_model", type=str, required=True, help="Environment model file")
    parser.add_argument("--train", type=str, required=True, help="Location of training data")
    parser.add_argument("--val", type=str, required=True, help="Location of validation data")
    parser.add_argument(
        "--per_eq_tol",
        type=float,
        required=True,
        help="Percent of latent state elements that need to be equal to declare equal",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--lr_d",
        type=float,
        default=0.9999993,
        help="Learning rate decay for every iteration. Learning rate is decayed according to: lr * (lr_d ^ itr)",
    )
    parser.add_argument("--max_itrs", type=int, default=1_000_000, help="Maximum number of training iterations")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument(
        "--update_nnet_batch_size",
        type=int,
        default=1000,
        help="Batch size of each nnet used for each process update. Make smaller if running out of memory.",
    )
    parser.add_argument(
        "--states_per_update",
        type=int,
        default=100_000,
        help="How many states to train on before checking if target network should be updated",
    )
    parser.add_argument(
        "--loss_thresh",
        type=float,
        default=0.05,
        help="When the loss falls below this value, the target network is updated to the current network.",
    )
    parser.add_argument(
        "--update_itrs",
        type=str,
        default="",
        help="Iterations at which to update. Last itr will be max_itrs. Update defaults to loss_thresh if empty.",
    )
    parser.add_argument(
        "--max_solve_steps",
        type=int,
        default=1,
        help="Number of steps to take when trying to solve training states with greedy best-first "
        "search (GBFS). Each state encountered when solving is added to the training set. "
        "Number of steps starts at 1 and is increased every update until the maximum number "
        "is reached. Increasing this number can make the cost-to-go function more robust by "
        "exploring more of the state space.",
    )
    parser.add_argument("--num_test", type=int, default=1000, help="Number of test states.")
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
    parser.add_argument("--nnet_name", type=str, required=True, help="Name of neural network")
    parser.add_argument("--save_dir", type=str, default="saved_heur_models", help="Directory to which to save model")
    parser.add_argument("--ring", type=int, default=3, help="Ring buffer slots")
    parser.add_argument("--amp", action="store_true", default=False, help="Enable automatic mixed precision")
    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile for the DQN model")
    parser.add_argument(
        "--compile_all_models",
        action="store_true",
        default=False,
        help="If set, apply torch.compile to all local (unsharded) model replicas used for data generation "
        "and evaluation (env_model_gen, dqn_gen, dqn_targ_gen, env_model_eval, dqn_eval). Does not recompile "
        "the sharded training DQN (use --compile for that).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debugging output")
    # Optional target update triggers (default behavior: improvement-only)
    parser.add_argument(
        "--target_update_use_schedule",
        action="store_true",
        default=False,
        help="If set, reaching an iteration in update_itrs also triggers a target network update.",
    )
    parser.add_argument(
        "--target_update_use_loss_thresh",
        action="store_true",
        default=False,
        help="If set (and no schedule trigger), a loss falling below --loss_thresh triggers a target update.",
    )
    # Set replay_max>0 to enable. Samples in each on_batch are augmented with up to replay_frac
    # proportion drawn uniformly from the replay buffer (FIFO reservoir capped at replay_max)
    parser.add_argument(
        "--replay_max", type=int, default=0, help="Maximum number of recent samples to keep for replay (0=disabled)"
    )
    parser.add_argument(
        "--replay_frac",
        type=float,
        default=0.5,
        help="Fraction of current batch size to sample from replay buffer when augmenting (ignored if replay_max=0)",
    )

    args = parser.parse_args()
    argsd = vars(args)
    if argsd["update_itrs"]:
        argsd["update_itrs"] = [int(float(x)) for x in argsd["update_itrs"].split(",")]
        argsd["max_itrs"] = argsd["update_itrs"][-1]
    else:
        argsd["update_itrs"] = []
    model_dir = f"{argsd['save_dir']}/{argsd['nnet_name']}"
    argsd.update({
        "model_dir": model_dir,
        "targ_dir": f"{model_dir}/target",
        "curr_dir": f"{model_dir}/current",
        "output_save_loc": f"{model_dir}/output.txt",
    })
    return argsd


@dataclass(frozen=True, slots=True)
class QLearningDistConfig:
    """Configuration for the distributed Q-learning training loop."""

    env: str
    env_model: str
    train: str
    val: str
    per_eq_tol: float
    lr: float = 1e-3
    lr_d: float = 0.9999993
    max_itrs: int = 1_000_000
    batch_size: int = 1000
    update_nnet_batch_size: int = 1000
    states_per_update: int = 100_000
    loss_thresh: float = 0.05
    update_itrs: list[int] | None = None
    max_solve_steps: int = 1
    num_test: int = 1000
    start_steps: int = 0
    goal_steps: int = 0
    nnet_name: str = ""
    save_dir: str = "saved_heur_models"
    ring: int = 3
    amp: bool = False
    compile: bool = False
    compile_all_models: bool = False
    seed: int | None = None
    debug: bool = False
    # Optional replay, disabled when replay_max<=0
    replay_max: int = 0
    replay_frac: float = 0.5
    # Optional target update triggers
    target_update_use_schedule: bool = False
    target_update_use_loss_thresh: bool = False


def set_seed(seed: int, rank: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed (int): The base seed to use.
        rank (int): The rank of the current process.
    """
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def save_checkpoint_dcp(module: nn.Module, shards_dir: str, full_path: str | None) -> None:
    """Save an FSDP2/DTensor model checkpoint using Distributed Checkpoint (DCP).

    This runs on all ranks and writes a sharded checkpoint under ``shards_dir``.
    Optionally consolidates to a single-file ``torch.save`` at ``full_path`` (rank0 only).
    """
    os.makedirs(shards_dir, exist_ok=True)
    # Build DCP-compatible state dict
    state = {"model": get_model_state_dict(module)}
    state_dict_saver.save(state, checkpoint_id=shards_dir)

    if full_path is not None and (not dist.is_initialized() or dist.get_rank() == 0):
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        dcp_to_torch_save(shards_dir, full_path)


def load_or_init_dqn(env: Environment, path: str) -> nn.Module:
    """Load a DQN from path or initialize a new one from the environment.

    Args:
        env (Environment): The environment to use for initialization.
        path (str): The path to the DQN model file.

    Returns:
        nn.Module: The loaded or initialized DQN model.
    """
    if os.path.isfile(path):
        return nnet_utils.load_nnet(path, env.get_dqn())
    return env.get_dqn()


def train_loop(argsd: dict[str, Any]) -> None:
    """Distributed Q-learning training loop.

    Args:
        argsd (dict[str, Any]): The runtime arguments.
    """
    # Init distributed
    env_vars = dist_utils.get_env_vars()
    dist_utils.setup_ddp(
        env_vars["master_addr"],
        env_vars["master_port"],
        env_vars["world_rank"],
        env_vars["world_size"],
        env_vars.get("local_rank"),
    )
    world_rank = env_vars["world_rank"] if dist.is_initialized() else 0
    world_size = env_vars["world_size"] if dist.is_initialized() else 1
    is_rank0 = world_rank == 0

    # Device already set inside setup_ddp, only set here if not yet selected
    if torch.cuda.is_available():
        want_dev = int(os.environ.get("LOCAL_RANK", world_rank % torch.cuda.device_count()))
        if torch.cuda.current_device() != want_dev:
            torch.cuda.set_device(want_dev)

    device = torch.device("cuda")
    if argsd["seed"] is not None:
        set_seed(argsd["seed"], world_rank)

    if is_rank0:
        os.makedirs(argsd["targ_dir"], exist_ok=True)
        os.makedirs(argsd["curr_dir"], exist_ok=True)
        with open(f"{argsd['model_dir']}/args.pkl", "wb") as f:
            pickle.dump(argsd, f, protocol=pickle.HIGHEST_PROTOCOL)

    writer: SummaryWriter | None = SummaryWriter(log_dir=argsd["model_dir"]) if is_rank0 else None
    if is_rank0 and not argsd["debug"] and not isinstance(sys.stdout, data_utils.Logger):
        sys.stdout = data_utils.Logger(argsd["output_save_loc"], "a")
    print_args(argsd)

    # Load environment & data
    env: Environment = env_utils.get_environment(argsd["env"])
    env_model_file = f"{argsd['env_model']}/env_state_dict.pt"
    # Load environment model, ensure it's on the CUDA device
    env_model: Any = nnet_utils.load_nnet(env_model_file, env.get_env_nnet(), device)
    env_model.to(device)
    # Offline train states (for generation) & validation state (fixed test set)
    with open(argsd["train"], "rb") as ftr, open(argsd["val"], "rb") as fval:
        episodes_tr = pickle.load(ftr)
        episodes_val = pickle.load(fval)

    states_train_np = np.concatenate(episodes_tr[0], axis=0)
    states_val_np = np.concatenate(episodes_val[0], axis=0)

    # Fixed evaluation sample for GBFS
    samp = np.random.randint(0, states_val_np.shape[0], size=argsd["num_test"])
    states_start_t_np = states_val_np[samp]
    goal_steps_samp = list(np.random.randint(0, argsd["goal_steps"] + 1, size=argsd["num_test"]))
    num_actions = env.num_actions_max
    assert num_actions is not None
    states_goal_t_np = imag_utils.random_walk(states_start_t_np, goal_steps_samp, num_actions, env_model, device)

    # Convert offline training pool to GPU tensor (uint8)
    offline_states_t = torch.tensor(states_train_np.astype(np.float32, copy=False), device=device)

    # Build meshes & shard models
    mesh_env: DeviceMesh = init_device_mesh("cuda", (world_size,))
    mesh_dqn: DeviceMesh = init_device_mesh("cuda", (world_size,))

    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    cast_forward_inputs: bool = True
    if argsd["amp"] and torch.cuda.is_bf16_supported():
        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32

    mp_env: MixedPrecisionPolicy = MixedPrecisionPolicy(param_dtype, reduce_dtype, output_dtype, cast_forward_inputs)
    mp_dqn: MixedPrecisionPolicy = MixedPrecisionPolicy(param_dtype, reduce_dtype, output_dtype, cast_forward_inputs)
    env_model = fully_shard(env_model, mesh=mesh_env, mp_policy=mp_env)
    env_model.eval()
    env_model.set_requires_gradient_sync(False, recurse=True)

    curr_path = f"{argsd['curr_dir']}/model_state_dict.pt"
    dqn: Any = load_or_init_dqn(env, curr_path)

    # Target network initialization
    targ_path = f"{argsd['targ_dir']}/model_state_dict.pt"
    if os.path.isfile(targ_path):
        dqn_targ: Any = load_or_init_dqn(env, targ_path)
    else:
        # Build a full DQN replica and zero out its parameters so
        # the architecture matches from the start
        dqn_targ = env.get_dqn()
        with torch.no_grad():
            for p in dqn_targ.parameters():
                p.zero_()

        if is_rank0:
            print("[Init] Created zero-initialized target DQN.")

    dqn = fully_shard(dqn, mesh=mesh_dqn, mp_policy=mp_dqn)
    dqn_targ = fully_shard(dqn_targ, mesh=mesh_dqn, mp_policy=mp_dqn)
    dqn_targ.eval()

    dqn_targ.set_requires_gradient_sync(False, recurse=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    if argsd["compile"]:
        try:
            dqn.compile(mode="default")
            env_model.compile(mode="default")
            print("[INFO] Compiled dqn for training.\n[INFO] Compiled env_model for training.")
        except Exception as e:
            if is_rank0:
                print(f"[WARN] torch.compile failed: {e}")

    # Optimizer / scheduling
    opt: Optimizer
    try:  # Attempt fused Adam (PyTorch 2.1+)
        opt = optim.Adam(dqn.parameters(), lr=argsd["lr"], fused=True)
        if is_rank0:
            print("Using fused Adam optimizer.")
    except Exception:
        opt = optim.Adam(dqn.parameters(), lr=argsd["lr"])

    # Iterative training state
    update_num = 0
    per_solved_best = 0.0
    # track how many train iterations have occurred since last target update (for schedule calc)
    # (we already recompute from itr and update_itrs diff, but keep clarity)
    global_batch = argsd["batch_size"]
    base_local = max(1, global_batch // world_size) if world_size > 0 else global_batch
    # Use a uniform per-rank base_local (drop remainder)
    local_batch = base_local
    effective_global_batch = local_batch * world_size if world_size > 0 else local_batch
    if effective_global_batch != global_batch and is_rank0:
        print(
            f"[INFO] Adjusted effective global batch from {global_batch} to {effective_global_batch} "
            f"(dropped remainder {global_batch - effective_global_batch}) for sync safety."
        )
    # ------------------------------------------------------------------
    # Micro-batch accumulation semantics:
    # In the ``qlearning.py`` implementation each optimizer iteration consumes exactly
    # 'batch_size' samples. 'update_nnet_batch_size' there only influences how
    # much data is generated per outer update (by enlarging generation batch),
    # not the logical iteration size. Here we instead accumulate gradients to
    # reach the logical per-iteration batch without increasing iteration span.
    #
    # We interpret update_nnet_batch_size as the desired (approximate) global
    # micro-batch size used for a single forward/backward prior to accumulation.
    # Therefore number of micro-steps per optimizer step is:
    #   micro_accum = ceil(global_batch / update_nnet_batch_size)
    # and the per-rank micro batch is local_batch / micro_accum (ceil).
    # If update_nnet_batch_size >= global_batch -> micro_accum == 1.
    # This keeps itr meaning consistent with qlearning.py
    # (one optimizer step == one logical "Itr" consuming ~global_batch samples total)
    # ------------------------------------------------------------------
    update_nnet_batch_size = max(1, argsd["update_nnet_batch_size"])
    micro_accum = max(1, int(np.ceil(global_batch / update_nnet_batch_size)))
    micro_local_batch = max(1, int(np.ceil(local_batch / micro_accum)))
    if is_rank0:
        print(
            f"Local batch {local_batch}, micro_local_batch {micro_local_batch}, micro_accum {micro_accum}, "
            f"effective_global_batch_per_itr ~{global_batch}, world_size {world_size}, ring={argsd['ring']}"
        )

    use_autocast = bool(argsd["amp"] and torch.cuda.is_bf16_supported())

    # Timing / progress (rank0 only prints)
    display_itrs = 100  # match style of qlearning.py
    times: OrderedDict[str, float] = OrderedDict([("fprop", 0.0), ("bprop", 0.0), ("itr", 0.0)])

    # Replay config
    replay_max: int = int(argsd.get("replay_max", 0) or 0)
    replay_frac: float = float(argsd.get("replay_frac", 0.5)) if replay_max > 0 else 0.0

    # Build the on_batch handler
    on_batch_handler = OnBatchHandler(
        dqn=dqn,
        opt=opt,
        device=device,
        world_size=world_size,
        lr=argsd["lr"],
        lr_d=argsd["lr_d"],
        max_itrs=argsd["max_itrs"],
        micro_local_batch=micro_local_batch,
        micro_accum=micro_accum,
        use_autocast=use_autocast,
        is_rank0=is_rank0,
        display_itrs=display_itrs,
        times=times,
        replay_max=replay_max,
        replay_frac=replay_frac,
    )

    # Main training loop
    try:
        while on_batch_handler.itr < argsd["max_itrs"]:
            max_steps = min(update_num + 1, argsd["max_solve_steps"])
            if len(argsd["update_itrs"]) > update_num:
                target_train_itrs = argsd["update_itrs"][update_num] - on_batch_handler.itr
            else:
                # Adjust for micro-accum so that logical iterations consume approx states_per_update samples
                effective_global_per_itr = global_batch  # by construction after refactor
                target_train_itrs = int(np.ceil(argsd["states_per_update"] / effective_global_per_itr))

            # Estimate number of generation batches (each may yield <= max_steps unroll steps)
            num_batches: int = int(np.ceil(target_train_itrs / max(1, max_steps)))
            if is_rank0:
                print(f"\nUpdate {update_num} target train itrs {target_train_itrs} (max_solve_steps={max_steps})")
                print(
                    "Generating data with local batch: "
                    f"{local_batch}, micro_local_batch: {micro_local_batch}, "
                    f"micro_accum: {micro_accum}, num_generation_batches: {num_batches}"
                )
            # ------------------------------------------------------------------
            # Freeze current policy for data generation (stationary snapshot):
            # We gather a full (unsharded) state dict from the possibly FSDP sharded
            # training DQN and load it into a *local* (unsharded) replica used only
            # for action selection & bootstrap targets during this update's data gen.
            # This matches the single-GPU workflow where data is generated before
            # any weight updates for that iteration
            # ------------------------------------------------------------------

            with torch.no_grad():
                policy_sd = get_model_state_dict(
                    dqn, options=StateDictOptions(full_state_dict=True, cpu_offload=False, broadcast_from_rank0=False)
                )
                dqn_gen = env.get_dqn()
                missing, unexpected = dqn_gen.load_state_dict(policy_sd)
                if is_rank0 and (missing or unexpected):
                    print(f"[WARN] Snapshot load issues. Missing: {missing}, Unexpected: {unexpected}")

                # Move snapshot to training device & align dtype with sharded training model (bf16 if AMP)
                try:
                    ref_dtype = next(dqn.parameters()).dtype
                except StopIteration:
                    ref_dtype = torch.float32

                dqn_gen.to(device=device, dtype=ref_dtype)
                dqn_gen.eval()

                # Optionally compile local generation replica
                if argsd["compile_all_models"]:
                    try:
                        dqn_gen.compile(mode="default")
                        if is_rank0:
                            print("[INFO] Compiled dqn_gen for data generation.")
                    except Exception as e:
                        if is_rank0:
                            print(f"[WARN] torch.compile failed for dqn_gen: {e}")

                if is_rank0 and argsd["debug"]:
                    print(
                        f"[DEBUG] Loaded DQN state dict for data generation. "
                        f"dqn_gen device: {next(dqn_gen.parameters()).device} dtype: {next(dqn_gen.parameters()).dtype}"
                    )

            gen_start_time = time.time()
            # -------------------------------------------------------------------------------
            # Build local (unsharded) replicas of env_model and target network for generation:
            # Generation may perform a *variable* number of forward passes per rank
            # due to early trajectory termination. If we used the sharded versions, differing
            # forward counts would desynchronize FSDP/NCCL collectives. By using local replicas
            # (no collectives) we can retain efficient early-exit logic and maximally utilize
            # each GPU while still trimming samples in on_batch() to keep training iterations
            # synchronized.
            # -------------------------------------------------------------------------------
            with torch.no_grad():
                # Snapshot env model
                env_model_sd = get_model_state_dict(
                    env_model,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=False, broadcast_from_rank0=False),
                )
                env_model_gen = env.get_env_nnet()
                miss_em, unexp_em = env_model_gen.load_state_dict(env_model_sd)

                if is_rank0 and (miss_em or unexp_em):
                    print(f"[WARN] Env model gen snapshot mismatch. missing={miss_em} unexpected={unexp_em}")

                try:
                    ref_env_dtype = next(env_model.parameters()).dtype
                except StopIteration:
                    ref_env_dtype = torch.float32

                env_model_gen.to(device=device, dtype=ref_env_dtype)
                env_model_gen.eval()

                if argsd["compile_all_models"]:
                    try:
                        env_model_gen.compile(mode="default")
                        if is_rank0:
                            print("[INFO] Compiled env_model_gen for data generation.")
                    except Exception as e:
                        if is_rank0:
                            print(f"[WARN] torch.compile failed for env_model_gen: {e}")

                # Snapshot target network
                dqn_targ_sd = get_model_state_dict(
                    dqn_targ,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=False, broadcast_from_rank0=False),
                )
                dqn_targ_gen = env.get_dqn()
                miss_tt, unexp_tt = dqn_targ_gen.load_state_dict(dqn_targ_sd)

                if is_rank0 and (miss_tt or unexp_tt):
                    print(f"[WARN] Target model gen snapshot mismatch. missing={miss_tt} unexpected={unexp_tt}")
                # Match device & dtype with live target (bf16 if AMP)
                try:
                    ref_targ_dtype = next(dqn_targ.parameters()).dtype
                except StopIteration:
                    ref_targ_dtype = torch.float32

                dqn_targ_gen.to(device=device, dtype=ref_targ_dtype)
                dqn_targ_gen.eval()

                if argsd["compile_all_models"]:
                    try:
                        dqn_targ_gen.compile(mode="default")
                        if is_rank0:
                            print("[INFO] Compiled dqn_targ_gen for data generation.")
                    except Exception as e:
                        if is_rank0:
                            print(f"[WARN] torch.compile failed for dqn_targ_gen: {e}")

            # Run asynchronous generation feeding on_batch with local replicas only
            update_utils_dist.generate_batches_gpu(
                env,
                env_model_gen,
                dqn_gen,  # frozen snapshot (unsharded)
                dqn_targ_gen,  # frozen target (unsharded)
                offline_states_t,
                batch_size=local_batch,
                num_batches=num_batches,
                start_steps=argsd["start_steps"],
                goal_steps=argsd["goal_steps"],
                per_eq_tol=argsd["per_eq_tol"],
                max_solve_steps=max_steps,
                temp=1 / 3.0,
                device=device,
                ring_size=argsd["ring"],
                on_batch=on_batch_handler,
                verbose=is_rank0,
                debug=argsd["debug"],
            )

            # Synchronize all CUDA work before distributed barrier
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Use monitored_barrier only if backend is gloo
            if dist.is_initialized():
                backend = dist.get_backend()
                if is_rank0:
                    print(
                        f"Generation + training time: {time.time() - gen_start_time:.2f}s"
                        "\n[Sync] Entering post-generation distributed barrier..."
                    )
                try:
                    mb = getattr(dist, "monitored_barrier", None)
                    if callable(mb) and backend == "gloo":
                        mb(timeout=datetime.timedelta(minutes=10))
                    else:
                        dist.barrier()
                except Exception as e:
                    if is_rank0:
                        print(f"[WARN] Barrier after generation fallback/issue: {e}")

                if is_rank0:
                    print("[Sync] All ranks passed post-generation barrier.")

            elif is_rank0:
                print(f"Generation + training time: {time.time() - gen_start_time:.2f}s (no dist initialized)")

            # Save current model
            save_start = time.time()
            if is_rank0:
                print("[Sync] Saving current model checkpoint (DCP sharded + optional consolidated)...")

            # Save on all ranks (DCP collectives)
            save_checkpoint_dcp(cast(nn.Module, dqn), f"{argsd['curr_dir']}/dcp_shards", curr_path)

            if is_rank0:
                print(f"[Sync] Model checkpoint saved in {time.time() - save_start:.2f}s")

            # Second barrier to ensure snapshot save complete before evaluation (other ranks idle waiting)
            if dist.is_initialized():
                try:
                    dist.barrier()  # Ensures broadcast later sees consistent file
                except Exception as e:
                    if is_rank0:
                        print(f"[WARN] Barrier before evaluation failed: {e}")

            # Capture update_num BEFORE potential rank0-only increment so all ranks (including rank0)
            # can detect the change after the broadcast. The previous implementation captured
            # prev_update_num AFTER rank0 had already incremented it, causing rank0 to skip
            # updating its in-memory target network parameters (only non-zero ranks updated),
            # leaving rank0's target stale and desynchronizing generation targets across ranks.
            prev_update_num_local = int(update_num)
            do_update: bool = False
            if dist.is_initialized():
                with contextlib.suppress(Exception):
                    dist.barrier()

            if is_rank0:
                print("\n[Eval] Starting GBFS evaluation...")
                # Load DQN weights from the saved checkpoint file
                writer = cast(SummaryWriter, writer)
                dqn_eval = env.get_dqn()
                checkpoint = torch.load(curr_path)
                # Handle the case where checkpoint is wrapped in a "model" key (from DCP save)
                dqn_eval.load_state_dict(
                    checkpoint["model"] if (isinstance(checkpoint, dict) and "model" in checkpoint) else checkpoint
                )
                dqn_eval.to(device)
                dqn_eval.eval()

                if argsd["debug"]:
                    print("\n[DEBUG] Loaded DQN for evaluation")

                # Plain env model replica from pretrained file
                env_model_eval = nnet_utils.load_nnet(env_model_file, env.get_env_nnet(), device)
                env_model_eval.eval()
                eval_start_ts = time.time()
                print(f"[Eval] GBFS fixed-set evaluation start | itr={on_batch_handler.itr} update={update_num}")
                max_gbfs_steps = min(update_num + 1, argsd["goal_steps"])
                print(f"\nTesting with {max_gbfs_steps} GBFS steps\nFixed test states ({states_start_t_np.shape[0]})")
                solved_mask, _ = gbfs(
                    dqn_eval,
                    env_model_eval,
                    states_start_t_np,
                    states_goal_t_np,
                    argsd["per_eq_tol"],
                    max_gbfs_steps,
                    device,
                )
                per_fixed = 100.0 * float(np.sum(solved_mask)) / len(solved_mask)
                prev_best = per_solved_best  # capture best before possible update
                improved_fixed = per_fixed > prev_best
                if improved_fixed:
                    per_solved_best = per_fixed

                # Logging
                print(
                    f"[Eval] Greedy policy solved (fixed set): {per_fixed:.2f}% | Prev Best: {prev_best:.2f}% - "
                    f"Improved: {improved_fixed}"
                )

                print("[Eval] Running gbfs_test (random sample evaluation)...")
                per_test = gbfs_test(
                    states_val_np,
                    argsd["num_test"],
                    dqn_eval,
                    env_model_eval,
                    num_actions,
                    argsd["goal_steps"],
                    device,
                    max_gbfs_steps,
                    argsd["per_eq_tol"],
                )
                writer.add_scalar("per_solved", per_test, on_batch_handler.itr)
                writer.add_scalar("loss_last", on_batch_handler.last_loss_val, on_batch_handler.itr)
                writer.flush()
                print(
                    f"[Eval] Completed in {time.time() - eval_start_ts:.2f}s. per_solved_test: {per_test:.2f} "
                    f"(loss_last={on_batch_handler.last_loss_val:.3E})"
                )
                # Target update policy:
                # Always allow GBFS improvement
                # Optionally allow schedule and/or loss threshold triggers if flags enabled
                reasons: list[str] = []
                do_update = False
                if improved_fixed:
                    do_update = True
                    reasons.append(f"gbfs_improve(per_fixed {per_fixed:.2f} > prev_best {prev_best:.2f})")

                elif (
                    argsd["target_update_use_schedule"]
                    and len(argsd["update_itrs"]) > update_num
                    and on_batch_handler.itr >= argsd["update_itrs"][update_num]
                ):
                    do_update = True
                    reasons.append("schedule")

                elif (
                    argsd["target_update_use_loss_thresh"]
                    and not argsd["target_update_use_schedule"]
                    and on_batch_handler.last_loss_val <= argsd["loss_thresh"]
                ):
                    do_update = True
                    reasons.append(f"loss {on_batch_handler.last_loss_val:.3E} <= {argsd['loss_thresh']:.3E}")

                if do_update:
                    reason_str = ", ".join(reasons)
                    print(f"[Target Update] Reasons: {reason_str} | update_num -> {update_num + 1}")
                    data_utils.copy_files(argsd["curr_dir"], argsd["targ_dir"])
                    update_num += 1

                with open(f"{argsd['curr_dir']}/status.pkl", "wb") as fout:
                    pickle.dump(
                        (on_batch_handler.itr, update_num, states_start_t_np, states_goal_t_np, per_solved_best),
                        fout,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                print(
                    f"Itr {on_batch_handler.itr}, "
                    f"last_loss {on_batch_handler.last_loss_val:.4E}, "
                    f"per_solved_test {per_test:.2f}, "
                    f"best_fixed {per_solved_best:.2f} update_num {update_num}"
                )

            # Sync after rank0-only evaluation to keep collective order aligned
            if dist.is_initialized():
                try:
                    dist.barrier()
                except Exception as e:
                    if is_rank0:
                        print(f"[WARN] Barrier after evaluation failed: {e}")

            # Broadcast the possibly incremented update_num from rank0
            bcast = torch.tensor([update_num], device=device, dtype=torch.int64)
            dist.broadcast(bcast, src=0)
            update_num = int(bcast.item())

            # If update_num changed relative to the pre-eval snapshot, update in-memory target
            if update_num != prev_update_num_local:
                with torch.no_grad():
                    # Use a sharded (per-rank) state dict, avoid unnecessary full gathers
                    dqn_sd = get_model_state_dict(
                        dqn,
                        options=StateDictOptions(full_state_dict=False, cpu_offload=False, broadcast_from_rank0=False),
                    )
                    miss_tt, unexp_tt = set_model_state_dict(dqn_targ, dqn_sd)

                if is_rank0:
                    print(
                        f"[WARN] Target model set_state mismatch. missing={miss_tt} unexpected={unexp_tt}"
                        if miss_tt or unexp_tt
                        else "[Target Update] In-memory target network synchronized across all ranks."
                    )

            # Ensure all ranks finish any target sync before next iteration
            if dist.is_initialized():
                with contextlib.suppress(Exception):
                    dist.barrier()

            torch.cuda.empty_cache()

            if on_batch_handler.itr >= argsd["max_itrs"]:
                break

        if is_rank0:
            print("[DONE] Training complete.")
            if writer is not None:
                writer.close()

    finally:
        # Ensure group is destroyed even on exceptions to avoid resource leakage
        if dist.is_initialized():
            with contextlib.suppress(Exception):
                dist.barrier()
                dist.destroy_process_group()


# PUBLIC API


def run_qlearning_dist(cfg: QLearningDistConfig) -> None:
    """Programmatic API wrapper around CLI main loop."""
    argsd: dict[str, Any] = {
        **asdict(cfg),
        "update_itrs": cfg.update_itrs or [],
        "model_dir": f"{cfg.save_dir}/{cfg.nnet_name}",
    }
    argsd.update({
        "targ_dir": f"{argsd['model_dir']}/target",
        "curr_dir": f"{argsd['model_dir']}/current",
        "output_save_loc": f"{argsd['model_dir']}/output.txt",
    })
    if argsd["update_itrs"]:
        argsd["max_itrs"] = argsd["update_itrs"][-1]
    train_loop(argsd)


# ENTRY POINT


def main() -> None:
    """CLI entry that launches the distributed training loop on each rank."""
    parser = ArgumentParser()
    argsd = parse_arguments(parser)
    train_loop(argsd)


if __name__ == "__main__":
    main()
