from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterable
import contextlib
import os
from pathlib import Path
import pickle
from random import choice
import shutil
import sys
from typing import IO, Any, TextIO

import numpy as np
from numpy import float32
from numpy.typing import NDArray


class Logger:
    """Mirror `stdout` writes to both the terminal and a log file.

    By default, only rank 0 will print to the terminal and write to the log file.
    Other ranks will be silent unless explicitly enabled.
    """

    __slots__: str | Iterable[str] = ("lock", "terminal", "log", "log_to_terminal", "log_to_file", "rank", "devnull")

    def __init__(
        self,
        filename: str | os.PathLike[str],
        mode: str = "a",
        rank: int = 0,
        *,
        log_to_terminal: bool | None = None,
        log_to_file: bool | None = None,
        encoding: str = "utf-8",
        create_dir: bool = True,
    ) -> None:
        """Initialize the logger.

        Args:
            filename: Path to the log file.
            mode: File open mode (e.g., "a", "w").
            rank: Process rank (0-based). By default only rank 0 logs/prints.
            log_to_terminal: Override terminal logging (defaults to rank == 0).
            log_to_file: Override file logging (defaults to rank == 0).
            encoding: Text encoding for the log file.
            create_dir: Whether to create parent directories if missing.
        """
        # Determine behavior per rank
        self.rank: int = int(rank)
        self.log_to_terminal: bool = bool(self.rank == 0) if log_to_terminal is None else bool(log_to_terminal)
        self.log_to_file: bool = bool(self.rank == 0) if log_to_file is None else bool(log_to_file)

        # Keep a handle to the current stdout (so `print` still works if this instance becomes sys.stdout)
        self.terminal: TextIO = sys.stdout

        # Open a sink for disabled paths (no-op writes without branching the call sites)
        self.devnull: IO[str] = open(os.devnull, "w", encoding=encoding)  # noqa: SIM115
        self.log: IO[str]
        # Prepare the log file handle (or a devnull sink if file logging is disabled)
        if self.log_to_file:
            path: Path = Path(filename)
            if create_dir:
                # Handle cases where filename has no parent (e.g., "run.txt")
                parent: Path = path.parent
                if str(parent) not in {"", "."}:
                    parent.mkdir(parents=True, exist_ok=True)
            self.log = open(path, mode, encoding=encoding)  # noqa: SIM115
        else:
            self.log = self.devnull

        if self.log_to_terminal:
            # Use absolute path string, even if file logging is off
            abs_path: str = os.path.abspath(str(filename))
            print(f"Logging output to: {abs_path}")

        # # Thread-safety for concurrent writes
        # self.lock: threading.Lock = threading.Lock()

    def write(self, message: str) -> None:
        """Write a message to terminal and log file, then flush."""
        # with self.lock:
        if self.log_to_terminal:
            self.terminal.write(message)
        if self.log_to_file:
            self.log.write(message)

        self.flush()

    def flush(self) -> None:
        """Flush both terminal and log streams for `sys.stdout` compatibility."""
        # with self.lock:
        if self.log_to_terminal:
            with contextlib.suppress(Exception):
                self.terminal.flush()
        if self.log_to_file:
            self.log.flush()

    def close(self) -> None:
        """Close the underlying log stream (if not devnull)."""
        # with self.lock:
        try:
            if self.log is not self.devnull and not self.log.closed:
                self.log.close()
        finally:
            pass


def print_args(args: Namespace | dict[str, Any]) -> None:
    """Prints the argument names and their values.

    Args:
        args (Namespace | dict[str, Any]): The arguments to print. Can be an argparse.Namespace or
            a dictionary.
    """
    if isinstance(args, Namespace):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        raise ValueError("Invalid argument type. Expected argparse.Namespace or dict[str, Any].")  # pyright: ignore[reportUnreachable]

    labels: list[str] = [f"--{k}:" for k in args_dict]
    max_label_len: int = max((len(lbl) for lbl in labels), default=0)

    s: str = ""
    label: str
    padding: str
    for k, v in args_dict.items():
        label = f"--{k}:"
        # pad so all values start at the same column
        padding = " " * (max_label_len - len(label) + 2)
        s += f"{label}{padding}{v}\n"

    print(f"\n{'-' * 36}\nArguments being used:\n{'-' * 36}\n" + s + f"{'-' * 36}\n")


def get_file_path_without_extension(file_path: str) -> str:
    """Gets the file path without its extension.

    Args:
        file_path (str): The full file path.

    Returns:
        str: The file path without the extension.
    """
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    directory = os.path.dirname(file_path)
    return os.path.join(directory, file_name)


def load_states_from_files(
    num_states: int, data_files: list[str], load_outputs: bool = False
) -> tuple[list[NDArray[float32]], NDArray[float32]]:
    """Loads states from a list of data files.

    Args:
        num_states (int): The number of states to load.
        data_files (list[str]): The list of data files to load from.
        load_outputs (bool, optional): Whether to load outputs as well. Defaults to False.

    Returns:
        tuple[List, NDArray]: A tuple containing the list of states and the numpy array of
            outputs.
    """
    states: list[NDArray[float32]] = []
    outputs_l = []
    while len(states) < num_states:
        data_file = choice(data_files)
        with open(data_file, "rb") as file:
            data = pickle.load(file)

        rand_idxs = np.random.choice(len(data["states"]), len(data["states"]), replace=False)
        num_samps: int = min(num_states - len(states), len(data["states"]))

        for idx in range(num_samps):
            rand_idx = rand_idxs[idx]
            states.append(data["states"][rand_idx])

        if load_outputs:
            for idx in range(num_samps):
                rand_idx = rand_idxs[idx]
                outputs_l.append(data["outputs"][rand_idx])

    outputs = np.array(outputs_l)
    outputs = np.expand_dims(outputs, 1)

    return states, outputs


def copy_files(src_dir: str, dest_dir: str) -> None:
    """Copies files from the source directory to the destination directory.

    Args:
        src_dir (str): The source directory.
        dest_dir (str): The destination directory.
    """
    src_files: list[str] = os.listdir(src_dir)
    for file_name in src_files:
        full_file_name: str = os.path.join(src_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest_dir)
