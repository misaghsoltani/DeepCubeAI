"""GPU-resident producer utilities (ring buffer) for Q-learning."""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np
import torch
from torch import Tensor, nn
import torch.distributed as dist

from deepcubeai.environments.environment_abstract import Environment


class _OnBatchLike(Protocol):
    itr: int
    max_itrs: int

    def __call__(self, states_start: Tensor, states_goal: Tensor, acts: Tensor, ctgs: Tensor) -> None: ...


def compute_loss(q_vals: Tensor, acts: Tensor, ctg_t: Tensor) -> tuple[Tensor, Tensor]:
    """Compute per-batch loss and return both loss and gathered action values.

    Returns:
        loss (Tensor): Scalar loss tensor.
        q_act (Tensor): Q-values for taken actions.
    """
    q_act: Tensor = q_vals.gather(1, acts.unsqueeze(1))[:, 0]
    diff: Tensor = q_act - ctg_t
    ad: Tensor = diff.abs()
    huber: Tensor = 0.5 * diff.pow(2) * (ad <= 1.0) + (ad - 0.5) * (ad > 1.0)
    loss: Tensor = (diff.pow(2) * (diff >= 0) + huber * (diff < 0)).mean()
    return loss, q_act


class ReplayBuffer:
    """Lightweight on-device replay buffer (FIFO + reservoir style trimming)."""

    @torch.no_grad()
    def __init__(self, capacity: int, frac: float) -> None:
        self.capacity: int = int(capacity)
        self.frac: float = float(frac) if capacity > 0 else 0.0
        self.states_start: list[Tensor] = []
        self.states_goal: list[Tensor] = []
        self.actions: list[Tensor] = []
        self.ctgs: list[Tensor] = []
        self.count: int = 0

    @torch.no_grad()
    def add(self, s_start: Tensor, s_goal: Tensor, acts: Tensor, ctgs: Tensor) -> None:
        """Append a new chunk to the buffer and trim if capacity exceeded."""
        if self.capacity <= 0:
            return
        self.states_start.append(s_start.detach())
        self.states_goal.append(s_goal.detach())
        self.actions.append(acts.detach())
        self.ctgs.append(ctgs.detach())
        self.count += s_start.shape[0]
        while self.count > self.capacity and self.states_start:
            removed = self.states_start.pop(0)
            self.states_goal.pop(0)
            self.actions.pop(0)
            self.ctgs.pop(0)
            self.count -= removed.shape[0]

    @torch.no_grad()
    def sample(self, num: int) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        """Uniformly sample up to `num` items from the concatenated buffer."""
        if self.capacity <= 0 or self.count == 0 or num <= 0:
            return None
        rs = torch.cat(self.states_start, dim=0)
        rg = torch.cat(self.states_goal, dim=0)
        ra = torch.cat(self.actions, dim=0)
        rc = torch.cat(self.ctgs, dim=0)
        num = min(num, rs.shape[0])
        perm = torch.randperm(rs.shape[0], device=rs.device)[:num]
        return rs[perm], rg[perm], ra[perm], rc[perm]


# Sampling / backup helpers


@torch.no_grad()
def sample_boltzmann(qvals: Tensor, temp: float) -> Tensor:
    """Sample actions from Boltzmann distribution over Q-values.

    Args:
        qvals: Tensor of Q-values for each action. Shape: (Batch size, number of actions).
        temp: Temperature parameter for Boltzmann distribution.

    Returns:
        Tensor of sampled actions.
    """
    exp_vals = torch.exp((1.0 / temp) * (-qvals + qvals.min(dim=1, keepdim=True)[0]))
    probs = exp_vals / exp_vals.sum(dim=1, keepdim=True)
    return torch.multinomial(probs, 1)[:, 0]


@torch.no_grad()
def q_step(
    states: Tensor,
    states_goal: Tensor,
    per_eq_tol: float,
    env_model: nn.Module,
    dqn: nn.Module,
    dqn_targ: nn.Module,
    temp: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Perform one Q-learning step: compute next states, actions, and backups.

    Args:
        states: Current states (B, D) tensor.
        states_goal: Goal states (B, D) tensor.
        per_eq_tol: Per-equation tolerance for success.
        env_model: Environment model (forward dynamics).
        dqn: Q-network (online).
        dqn_targ: Q-network (target).
        temp: Temperature for action sampling.

    Returns:
        next_states: Next states after applying actions (B, D) tensor.
        acts: Sampled actions (B,) tensor.
        backups: Backups for the next states (B,) tensor.
        is_solved: Boolean tensor indicating if each state is solved (B,).
    """
    qvals = dqn(states, states_goal)
    acts = sample_boltzmann(qvals, temp)
    is_solved = (100 * torch.mean(torch.eq(states, states_goal).float(), dim=1)) >= per_eq_tol
    next_states = env_model(states, acts).round()
    ctg_acts_next = torch.clamp(dqn_targ(next_states, states_goal), min=0)
    ctgs_next = torch.min(ctg_acts_next, dim=1)[0]
    backups = (1.0 + ctgs_next) * (1.0 - is_solved.float())
    return next_states, acts, backups, is_solved


# Ring buffer generation


class RingSlot:
    """One ring buffer slot living fully on device."""

    def __init__(self) -> None:  # noqa: D401
        self.states_start: Tensor | None = None
        self.states_goal: Tensor | None = None
        self.actions: Tensor | None = None
        self.ctgs: Tensor | None = None
        self.ready_ev: torch.cuda.Event = torch.cuda.Event()
        self.consumed_ev: torch.cuda.Event = torch.cuda.Event()


@torch.no_grad()
def random_walk_gpu(base_states: Tensor, num_actions: int, steps_list: list[int], env_model: nn.Module) -> Tensor:
    """Perform random walks on GPU using env_model.

    Args:
        base_states: (B, D) uint8/float tensor
        num_actions: max number of actions
        steps_list: list of ints length B
        env_model: sharded env model (eval)

    Returns:
        Tensor of shape (B, D) with resulting states (float32)
    """
    device = base_states.device
    # Match env_model parameter dtype (may be bf16 under mixed precision)
    try:
        param_dtype = next(env_model.parameters()).dtype
    except StopIteration:
        param_dtype = base_states.dtype
    states = base_states.to(dtype=param_dtype)
    max_steps = max(steps_list) if steps_list else 0
    if max_steps == 0:
        return states
    steps_tensor = torch.tensor(steps_list, device=device)
    for step in range(max_steps):
        active = steps_tensor > step
        if not torch.any(active):
            break
        batch_idx = active.nonzero(as_tuple=False)[:, 0]
        # sample random actions uniformly
        acts = torch.randint(0, num_actions, (batch_idx.shape[0],), device=device)
        next_states = env_model(states[batch_idx], acts).round()
        if next_states.dtype != states.dtype:
            next_states = next_states.to(states.dtype)
        states[batch_idx] = next_states
    return states


def generate_batches_gpu(
    env: Environment,
    env_model: nn.Module,
    dqn: nn.Module,
    dqn_targ: nn.Module,
    offline_states: Tensor,
    *,
    batch_size: int,
    num_batches: int,
    start_steps: int,
    goal_steps: int,
    per_eq_tol: float,
    max_solve_steps: int,
    temp: float,
    device: torch.device,
    ring_size: int = 3,
    stream_gen: torch.cuda.Stream | None = None,
    stream_train: torch.cuda.Stream | None = None,
    on_batch: _OnBatchLike | None = None,
    verbose: bool = False,
    debug: bool = False,
) -> None:
    """Generate Q-learning training samples asynchronously and invoke callback.

    - All tensors remain on the specified CUDA device.
    - Ring buffer + Events ensure overlap of generation and consumption.
    """
    assert device.type == "cuda", "GPU-only pipeline required"
    num_actions = env.num_actions_max
    assert num_actions is not None

    # Streams
    if stream_gen is None:
        stream_gen = torch.cuda.Stream(priority=0)
    if stream_train is None:
        stream_train = torch.cuda.Stream(priority=0)

    # Ring buffer setup
    ring: list[RingSlot] = [RingSlot() for _ in range(ring_size)]
    # Mark all consumed initially so producer can start
    for rs in ring:
        rs.consumed_ev.record(torch.cuda.current_stream())

    # Precompute reused shapes lazily
    offline_n = offline_states.shape[0]
    slot_idx = 0

    # Determine progress batch indices (same set for all ranks to keep collective timing aligned)
    display_steps = set()
    if num_batches > 0:
        display_steps = set(np.linspace(1, num_batches, 10, dtype=int).tolist())

    start_time = time.time()
    total_samples = 0  # per-rank sample count
    for batch_idx in range(1, num_batches + 1):
        # Early stop: if consumer reports having reached max iterations, stop generating
        if on_batch is not None and on_batch.itr >= on_batch.max_itrs:
            break

        rs = ring[slot_idx]

        # Generator (stream_gen)

        with torch.cuda.stream(stream_gen), torch.no_grad():
            stream_gen.wait_event(rs.consumed_ev)  # pyright: ignore[reportArgumentType]
            samp = torch.randint(0, offline_n, (batch_size,), device=device)
            start_steps_list = [start_steps] * batch_size
            goal_steps_list = torch.randint(0, goal_steps + 1, (batch_size,), device=device)

            base_states = offline_states[samp].to(device=device, dtype=torch.float32, non_blocking=True)
            states_start = random_walk_gpu(base_states, num_actions, start_steps_list, env_model)
            states_goal = random_walk_gpu(states_start.clone(), num_actions, goal_steps_list.tolist(), env_model)

            states_start = states_start.contiguous()
            states_goal = states_goal.contiguous()

            # For each solve step:
            # We'll run sequentially but still inside generation stream so training waits
            states_curr = states_start
            states_goal_curr = states_goal
            all_states_start: list[Tensor] = []
            all_states_goal: list[Tensor] = []
            all_actions: list[Tensor] = []
            all_ctgs: list[Tensor] = []
            for _step in range(max_solve_steps):
                next_states, acts, ctgs, solved = q_step(
                    states_curr, states_goal_curr, per_eq_tol, env_model, dqn, dqn_targ, temp
                )
                all_states_start.append(states_curr)
                all_states_goal.append(states_goal_curr)
                all_actions.append(acts)
                all_ctgs.append(ctgs)

                remaining_mask = torch.logical_not(solved)
                if not torch.any(remaining_mask):
                    break

                states_curr = next_states[remaining_mask]
                states_goal_curr = states_goal_curr[remaining_mask]

            rs.states_start = torch.cat(all_states_start, dim=0)
            rs.states_goal = torch.cat(all_states_goal, dim=0)
            rs.actions = torch.cat(all_actions, dim=0)
            rs.ctgs = torch.cat(all_ctgs, dim=0)

            # Ensure total count divisible by batch_size (round up)
            total_unroll = rs.states_start.size(0)
            rem = total_unroll % batch_size
            if rem != 0 and total_unroll > 0:
                need = batch_size - rem
                pad_idx = torch.randint(0, total_unroll, (need,), device=device)
                rs.states_start = torch.cat([rs.states_start, rs.states_start[pad_idx]], dim=0)
                rs.states_goal = torch.cat([rs.states_goal, rs.states_goal[pad_idx]], dim=0)
                rs.actions = torch.cat([rs.actions, rs.actions[pad_idx]], dim=0)
                rs.ctgs = torch.cat([rs.ctgs, rs.ctgs[pad_idx]], dim=0)
                if debug:
                    print(
                        "[DEBUG] Padded unroll from "
                        f"{total_unroll} to {total_unroll + need} (added {need}) "
                        f"for divisibility by {batch_size}."
                    )
            rs.ready_ev.record(stream_gen)

        # Consumer (stream_train)

        if on_batch is not None:
            with torch.cuda.stream(stream_train):
                stream_train.wait_event(rs.ready_ev)  # pyright: ignore[reportArgumentType]
                # Guard against BatchNorm (or other stat layers) receiving a batch of size 1
                if (
                    rs.states_start is None
                    or rs.states_start.size(0) < 2
                    or rs.states_goal is None
                    or rs.actions is None
                    or rs.ctgs is None
                ):
                    # Still mark consumed so producer advances
                    rs.consumed_ev.record(stream_train)
                else:
                    on_batch(rs.states_start, rs.states_goal, rs.actions, rs.ctgs)
                    rs.consumed_ev.record(stream_train)

        # Stats
        if rs.states_start is not None:
            total_samples += rs.states_start.shape[0]

        # Progress display (host side, rank0 should wrap outer call)
        # Compute global samples periodically (all ranks participate to avoid deadlock)
        global_samples = total_samples
        do_progress = batch_idx in display_steps or batch_idx == num_batches
        if do_progress and dist.is_available() and dist.is_initialized():
            try:
                ts = torch.tensor([total_samples], device=device, dtype=torch.int64)
                dist.all_reduce(ts, op=dist.ReduceOp.SUM)
                global_samples = int(ts.item())
            except Exception:
                pass
        if verbose and do_progress and batch_idx in display_steps:
            elapsed = time.time() - start_time
            pct = 100.0 * batch_idx / num_batches
            print(
                f"{pct:.2f}% ({elapsed:.2f}s) - gen batches: {batch_idx}/{num_batches}, "
                f"samples(local): {total_samples}, samples(global): {global_samples}"
            )

        slot_idx = (slot_idx + 1) % ring_size

    # Ensure all work done before returning
    torch.cuda.current_stream().wait_stream(stream_gen)
    torch.cuda.current_stream().wait_stream(stream_train)
    torch.cuda.synchronize()
    # Final global samples (all ranks call all_reduce if initialized)
    global_total = total_samples
    if dist.is_available() and dist.is_initialized():
        try:
            ts = torch.tensor([total_samples], device=device, dtype=torch.int64)
            dist.all_reduce(ts, op=dist.ReduceOp.SUM)
            global_total = int(ts.item())
        except Exception:
            pass

    if verbose:
        elapsed = time.time() - start_time
        print(f"Generated Samples (per-rank/global): {total_samples:,}/{global_total:,} - Time: {elapsed:.2f}s\n")
