import os
import pickle
import shutil
import sys
from argparse import Namespace
from random import choice
from typing import Any, Dict, List, Tuple

import numpy as np


class Logger:

    def __init__(self, filename: str, mode: str = "a"):
        """Initializes the Logger class.

        Args:
            filename (str): The name of the file to log to.
            mode (str, optional): The mode in which to open the file. Defaults to "a".
        """
        log_dir = os.path.dirname(filename)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.terminal = sys.stdout
        self.log = open(filename, mode, encoding="utf-8")

    def write(self, message: str) -> None:
        """Writes a message to both the terminal and the log file.

        Args:
            message (str): The message to write.
        """
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        """Flush method for compatibility with the `sys.stdout` interface."""


def print_args(args: Namespace | Dict[str, Any]) -> None:
    """Prints the argument names and their values.

    Args:
        args (Namespace | Dict[str, Any]): The arguments to print. Can be an argparse.Namespace or
            a dictionary.
    """
    if isinstance(args, Namespace):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        raise ValueError("Invalid argument type. Expected argparse.Namespace or dict[str, Any].")
    print("\n--------------------------------------------")
    print("Arguments being used:")
    print("--------------------------------------------")
    for arg_name, arg_value in args_dict.items():
        print(f"--{arg_name}:\t\t{arg_value}")
    print("--------------------------------------------\n")


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


def load_states_from_files(num_states: int,
                           data_files: List[str],
                           load_outputs: bool = False) -> Tuple[List, np.ndarray]:
    """Loads states from a list of data files.

    Args:
        num_states (int): The number of states to load.
        data_files (List[str]): The list of data files to load from.
        load_outputs (bool, optional): Whether to load outputs as well. Defaults to False.

    Returns:
        Tuple[List, np.ndarray]: A tuple containing the list of states and the numpy array of
            outputs.
    """
    states = []
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
    src_files: List[str] = os.listdir(src_dir)
    for file_name in src_files:
        full_file_name: str = os.path.join(src_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest_dir)
