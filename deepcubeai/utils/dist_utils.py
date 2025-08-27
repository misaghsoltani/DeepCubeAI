from __future__ import annotations

import contextlib
import os
from typing import Any

import torch
from torch.distributed import init_process_group


def get_env_vars() -> dict[str, Any]:
    """Get distributed environment variables.

    Priority order:
      1. torchrun / torch.distributed.run (RANK, WORLD_SIZE, LOCAL_RANK)
      2. SLURM (SLURM_PROCID, SLURM_NTASKS, LOCAL_RANK or derive)
      3. OpenMPI (OMPI_COMM_WORLD_*)
    Falls back to single-process defaults.
    """
    # Defaults
    world_rank = 0
    world_size = 1
    local_rank = 0
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", os.environ.get("PMI_PORT", "29500")))

    # torchrun style
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))

    # SLURM style
    elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        world_rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        # Derive local rank from GPUs per node if available
        gpus_per_node = os.environ.get("SLURM_GPUS_ON_NODE") or os.environ.get("SLURM_GPUS")
        if gpus_per_node:
            try:
                # If format like "2" or "gpu:2" keep digits only
                gpn_int = int("".join(ch for ch in gpus_per_node if ch.isdigit()) or "0")
                if gpn_int > 0:
                    local_rank = world_rank % gpn_int
            except ValueError:
                pass

    # OpenMPI style
    elif "OMPI_COMM_WORLD_SIZE" in os.environ:
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", world_size))
        world_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", world_rank))
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", local_rank))

    return {
        "world_rank": world_rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "master_addr": master_addr,
        "master_port": master_port,
    }


def get_init_method_str(master_addr: str, master_port: int) -> str:
    """Generate initialization method string for distributed processing.

    Args:
        master_addr (str): Master address.
        master_port (int): Master port.

    Returns:
        str: Initialization method string.
    """
    return f"tcp://{master_addr}:{master_port}"


def setup_ddp(master_addr: str, master_port: int, rank: int, world_size: int, local_rank: int | None = None) -> None:
    """Initialize ``torch.distributed`` process group with explicit device selection."""
    os.environ.setdefault("MASTER_ADDR", str(master_addr))
    os.environ.setdefault("MASTER_PORT", str(master_port))
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if world_size <= 1 or torch.distributed.is_initialized():
        return

    # Determine local rank (may come from env if not provided)
    if local_rank is None:
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        except ValueError:
            local_rank = 0

    # Set device early so process group init can use it
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())

    init_kwargs: dict[str, Any] = {
        "backend": backend,
        "rank": rank,
        "world_size": world_size,
        "init_method": get_init_method_str(master_addr, master_port),
    }
    # PyTorch 2.3+ supports device_id for NCCL collectives to suppress warning
    if backend == "nccl" and hasattr(torch.distributed.distributed_c10d, "_rank_not_in_group"):
        # Add device_id if current version supports it
        with contextlib.suppress(Exception):
            init_kwargs["device_id"] = torch.cuda.current_device()

    try:
        init_process_group(**init_kwargs)
    except TypeError:
        # Retry without device_id for older versions
        init_kwargs.pop("device_id", None)
        init_process_group(**init_kwargs)
