import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def flatten(data: List[List[Any]]) -> Tuple[List[Any], List[int]]:
    """
    Flattens a list of lists into a single list and returns the flattened list along with the
        split indices.

    Args:
        data (List[List[Any]]): The list of lists to be flattened.

    Returns:
        Tuple[List[Any], List[int]]: A tuple containing the flattened list and the split indices.
    """
    num_each = [len(x) for x in data]
    split_idxs: List[int] = list(np.cumsum(num_each)[:-1])

    data_flat = [item for sublist in data for item in sublist]

    return data_flat, split_idxs


def unflatten(data: List[Any], split_idxs: List[int]) -> List[List[Any]]:
    """
    Unflattens a list into a list of lists using the provided split indices.

    Args:
        data (List[Any]): The flattened list.
        split_idxs (List[int]): The indices to split the flattened list.

    Returns:
        List[List[Any]]: The unflattened list of lists.
    """
    data_split: List[List[Any]] = []

    start_idx: int = 0
    end_idx: int
    for end_idx in split_idxs:
        data_split.append(data[start_idx:end_idx])
        start_idx = end_idx

    data_split.append(data[start_idx:])

    return data_split


def split_evenly(num_total: int, num_splits: int) -> List[int]:
    """
    Splits a total number into nearly equal parts.

    Args:
        num_total (int): The total number to be split.
        num_splits (int): The number of parts to split into.

    Returns:
        List[int]: A list containing the sizes of each part.
    """
    num_per: List[int] = [math.floor(num_total / num_splits) for _ in range(num_splits)]
    left_over: int = num_total % num_splits
    for idx in range(left_over):
        num_per[idx] += 1

    return num_per


# Time profiling


def record_time(times: Dict[str, float], time_name: str, start_time: float, on_gpu: bool) -> None:
    """
    Records the elapsed time for a given time name and updates the times dictionary.
    Increments time if time_name is already in times. Synchronizes if on_gpu is true.

    Args:
        times (Dict[str, float]): The dictionary to store the times.
        time_name (str): The name of the time entry.
        start_time (float): The start time to calculate the elapsed time.
        on_gpu (bool): Whether to synchronize with GPU before recording time.
    """
    if on_gpu:
        torch.cuda.synchronize()

    time_elapsed = time.time() - start_time
    if time_name in times.keys():
        times[time_name] += time_elapsed
    else:
        times[time_name] = time_elapsed


def add_times(times: Dict[str, float], times_to_add: Dict[str, float]) -> None:
    """
    Adds times from one dictionary to another.

    Args:
        times (Dict[str, float]): The dictionary to update with added times.
        times_to_add (Dict[str, float]): The dictionary containing times to add.
    """
    for key, value in times_to_add.items():
        times[key] += value


def get_time_str(times: Dict[str, float]) -> str:
    """
    Converts a dictionary of times into a formatted string.

    Args:
        times (Dict[str, float]): The dictionary containing time entries.

    Returns:
        str: A formatted string representation of the times.
    """
    time_str_l: List[str] = []
    for key, val in times.items():
        time_str_i = f"{key}: {val:.2f}"
        time_str_l.append(time_str_i)
    time_str: str = ", ".join(time_str_l)

    return time_str
