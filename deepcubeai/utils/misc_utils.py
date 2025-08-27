from __future__ import annotations

import math
import os
import time
from typing import Any, TypeVar, cast, overload

import cv2
import numpy as np
from numpy.typing import NDArray
import torch

DT = TypeVar("DT", np.uint8, np.float32)


def flatten(data: list[list[Any]]) -> tuple[list[Any], list[int]]:
    """Flattens a list of lists into a single list and returns the flattened list along with the split indices.

    Args:
        data (list[list[Any]]): The list of lists to be flattened.

    Returns:
        tuple[list[Any], list[int]]: A tuple containing the flattened list and the split indices.
    """
    num_each: list[int] = [len(x) for x in data]
    split_idxs: list[int] = list(np.cumsum(num_each)[:-1])

    data_flat: list[Any] = [item for sublist in data for item in sublist]

    return data_flat, split_idxs


def unflatten(data: list[Any], split_idxs: list[int]) -> list[list[Any]]:
    """Unflattens a list into a list of lists using the provided split indices.

    Args:
        data (list[Any]): The flattened list.
        split_idxs (list[int]): The indices to split the flattened list.

    Returns:
        list[list[Any]]: The unflattened list of lists.
    """
    data_split: list[list[Any]] = []

    start_idx: int = 0
    end_idx: int
    for end_idx in split_idxs:
        data_split.append(data[start_idx:end_idx])
        start_idx = end_idx

    data_split.append(data[start_idx:])

    return data_split


def split_evenly(num_total: int, num_splits: int) -> list[int]:
    """Splits a total number into nearly equal parts.

    Args:
        num_total (int): The total number to be split.
        num_splits (int): The number of parts to split into.

    Returns:
        list[int]: A list containing the sizes of each part.
    """
    num_per: list[int] = [math.floor(num_total / num_splits) for _ in range(num_splits)]
    left_over: int = num_total % num_splits
    for idx in range(left_over):
        num_per[idx] += 1

    return num_per


# Time profiling


def record_time(times: dict[str, float], time_name: str, start_time: float, on_gpu: bool) -> None:
    """Records the elapsed time for a given time name and updates the times dictionary.

    Increments time if time_name is already in times. Synchronizes if on_gpu is true.

    Args:
        times (dict[str, float]): The dictionary to store the times.
        time_name (str): The name of the time entry.
        start_time (float): The start time to calculate the elapsed time.
        on_gpu (bool): Whether to synchronize with GPU before recording time.
    """
    if on_gpu:
        torch.cuda.synchronize()

    time_elapsed = time.time() - start_time
    if time_name in times:
        times[time_name] += time_elapsed
    else:
        times[time_name] = time_elapsed


def add_times(times: dict[str, float], times_to_add: dict[str, float]) -> None:
    """Adds times from one dictionary to another.

    Args:
        times (dict[str, float]): The dictionary to update with added times.
        times_to_add (dict[str, float]): The dictionary containing times to add.
    """
    for key, value in times_to_add.items():
        times[key] += value


def get_time_str(times: dict[str, float]) -> str:
    """Converts a dictionary of times into a formatted string.

    Args:
        times (dict[str, float]): The dictionary containing time entries.

    Returns:
        str: A formatted string representation of the times.
    """
    time_str_l: list[str] = []
    for key, val in times.items():
        time_str_i = f"{key}: {val:.2f}"
        time_str_l.append(time_str_i)
    time_str: str = ", ".join(time_str_l)

    return time_str


@overload
def imread_cv2(path: str, *, dtype: type[np.uint8]) -> NDArray[np.uint8]: ...


@overload
def imread_cv2(path: str, *, dtype: type[np.float32]) -> NDArray[np.float32]: ...


def imread_cv2(
    path: str, *, dtype: type[np.uint8] | type[np.float32] = np.uint8
) -> NDArray[np.uint8] | NDArray[np.float32]:
    """Read an image from file (first frame only), with channel order fix and optional dtype.

    - Uses IMREAD_UNCHANGED to preserve original bit depth & number of channels.
    - Converts BGR->RGB and BGRA->RGBA so 3/4-channel images come back RGB/RGBA.
    - Leaves grayscale as (H, W) with no extra channel.
    - For animated/multi-page formats, returns a single decoded image (use
      cv2.imreadmulti / cv2.imreadanimation for more frames).
    - Raises FileNotFoundError if the path doesn't exist.
    - Raises OSError if decoding fails.

    Dtype behavior:
        dtype=uint8   -> returns uint8. If source is 16-bit, scales to 0-255 via /257. If source is float32,
                            clips to [0,1] then scales by 255.
        dtype=float32 -> returns float32. If source is uint8/uint16, normalizes to [0,1] by dividing by 255
                            or 65535 respectively. If already float32, returned as-is.

    Args:
        path: Path to the image file.
        dtype: Desired output dtype, either uint8 (default) or float32.

    Returns:
        np.NDArraywith shape (H, W[, C]) and dtype matching `dtype`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: '{path}'")

    # Decode with original depth & channels
    img = cv2.imread(filename=path, flags=cv2.IMREAD_UNCHANGED)
    if img is None:
        # OpenCV returns an empty Mat/None on failure
        raise OSError(f"Failed to decode image: '{path}'")

    # Ensure grayscale stays 2D; only reorder for 3/4 channels
    if img.ndim == 3:
        c = img.shape[2]
        if c == 3:
            img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        elif c == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        # Uncommon channel counts (e.g., 2) are left as-is.

    # Convert dtype
    src_dtype = img.dtype

    # Handle float32 first to avoid type-checker thinking the later branch is unreachable
    if dtype is np.float32:
        if src_dtype == np.uint8:
            return cast(NDArray[np.float32], img.astype(np.float32) / 255.0)
        if src_dtype == np.uint16:
            return cast(NDArray[np.float32], img.astype(np.float32) / 65535.0)
        if src_dtype == np.float32:
            return cast(NDArray[np.float32], img.astype(np.float32))
        return cast(NDArray[np.float32], img.astype(np.float32))

    elif dtype is np.uint8:
        if src_dtype == np.uint8:
            return cast(NDArray[np.uint8], img)
        if src_dtype == np.uint16:
            # Map 0..65535 -> 0..255
            return cast(NDArray[np.uint8], (img.astype(np.uint32) // 257).astype(np.uint8))
        if src_dtype == np.float32:
            # Assume 0..1 float inputs (clip then scale)
            clipped = np.clip(img, 0.0, 1.0)
            return cast(NDArray[np.uint8], (clipped * 255.0 + 0.5).astype(np.uint8))
        # Fallback: clip to 0..255
        return cast(NDArray[np.uint8], np.clip(img, 0, 255).astype(np.uint8))

    # For unexpected dtype
    raise TypeError("dtype must be either uint8 or float32")
