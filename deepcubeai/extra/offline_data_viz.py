from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any, cast

import cv2
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from numpy import float32, intp
from numpy.typing import NDArray

from deepcubeai.utils.data_utils import print_args


def save_image(image_np: NDArray[float32], save_path: str) -> None:
    """Saves a NumPy array as an image file.

    Args:
        image_np (np.NDArray): The image data as a NumPy array of shape (C, H, W).
        save_path (str): The path where the image will be saved.
    """
    if image_np.ndim != 3:
        raise ValueError(f"Expected (C, H, W), got shape {image_np.shape}")

    c, _h, _w = image_np.shape
    img_u8 = (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)

    # CHW -> HWC
    img_hwc = np.transpose(img_u8, (1, 2, 0))

    # Channel handling for OpenCV's expected order
    img_out: NDArray[np.uint8]
    if c == 1:
        img_out = img_hwc[:, :, 0]  # (H, W) grayscale
    elif c == 3:
        img_out = cast(NDArray[np.uint8], cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR))  # RGB -> BGR
    elif c == 4:
        img_out = cast(NDArray[np.uint8], cv2.cvtColor(img_hwc, cv2.COLOR_RGBA2BGRA))  # RGBA -> BGRA
    else:
        raise ValueError(f"Unsupported channel count C={c}; expected 1, 3, or 4")

    ok: bool = cv2.imwrite(save_path, img_out)
    if not ok:
        # cv2.imwrite returns False on failure (bad path/unsupported extension/etc.)
        raise OSError(f"Failed to write image to '{save_path}'")


def plot_images(
    images_np: NDArray[float32],
    save_dir: str,
    action_episodes: NDArray[intp],
    episode_idxs: NDArray[intp],
    step_idxs: NDArray[intp],
    env: str,
) -> None:
    """Plots and saves images from episodes and steps.

    Args:
        images_np (np.NDArray): The image data as a NumPy array.
        save_dir (str): The directory where the images will be saved.
        action_episodes (np.NDArray): The actions taken in each episode.
        episode_idxs (np.NDArray): The indices of the episodes.
        step_idxs (np.NDArray): The indices of the steps within each episode.
        env (str): The environment name.
    """
    k = 0
    flag: bool = False
    flattened_shape = np.prod(step_idxs.shape)
    if images_np.shape[2] == 6:
        images_np = np.concatenate((images_np[:, :, :3], images_np[:, :, 3:]), axis=-1)

    ratio = 1  # images_np.shape[-1] / images_np.shape[-2]

    for i in range(len(episode_idxs)):
        for j in range(len(step_idxs[i]) - 1):
            if step_idxs[i, j + 1] == step_idxs[i, j] + 1:
                save_image(images_np[i, j], f"{save_dir}/{k}_{env}_seq{episode_idxs[i]}_step{step_idxs[i, j]}.png")
                save_image(
                    images_np[i, j + 1], f"{save_dir}/{k}_{env}_seq{episode_idxs[i]}_step{step_idxs[i, j + 1]}.png"
                )
                k += 1

                # Calculate progress percentage
                progress_percentage = (k / flattened_shape) * 100
                if progress_percentage % 10 == 0:
                    print(f"Progress: {progress_percentage:.2f}%")

                flag = True
                break

        if flag:
            break

    for i in range(images_np.shape[0]):
        _fig, ax_or_axes = plt.subplots(1, images_np.shape[1], figsize=(images_np.shape[1] * 2 * ratio, 2))
        # Normalize axes to a flat list
        axes_list: list[Axes]
        if images_np.shape[1] == 1:
            axes_list = [ax_or_axes]
        else:
            axes_nd = cast(np.ndarray, ax_or_axes)
            axes_list = [cast(Axes, ax) for ax in axes_nd.ravel().tolist()]

        for j in range(images_np.shape[1]):
            if step_idxs[i, j] == 0:
                action_taken = "None"
            else:
                action_taken = str(action_episodes[episode_idxs[i], step_idxs[i, j] - 1])

            ax = axes_list[j]
            ax.imshow(images_np[i, j].transpose(1, 2, 0))
            # ax.set_xticks(np.arange(0, images_np.shape[-1], 10))
            # ax.set_yticks(np.arange(0, images_np.shape[-2], 10))
            ax.tick_params(axis="both", which="both", labelsize=6)
            ax.set_title(
                f"Episode {episode_idxs[i]}, Step {step_idxs[i, j]}\nAction Taken: {action_taken}", size=8, pad=5
            )

        plt.savefig(f"{save_dir}/e{episode_idxs[i]}.png", dpi=300)
        # plt.show()
        plt.close()


@dataclass(frozen=True, slots=True)
class OfflineDataVizConfig:
    """Config for rendering sample images from offline datasets."""

    env: str
    train_data: str
    val_data: str
    num_train_trajs: int = 100
    num_train_steps: int = 100
    num_val_trajs: int = 100
    num_val_steps: int = 100
    save_imgs: str = "sample_images"

    @staticmethod
    def from_json(path: str) -> OfflineDataVizConfig:
        """Load config from JSON file, ignoring unknown keys."""
        raw: dict[str, Any] = json.loads(Path(path).read_bytes())
        return OfflineDataVizConfig(**raw)


def parse_arguments(parser: ArgumentParser) -> dict[str, Any]:
    """Parses command-line arguments.

    Args:
        parser (ArgumentParser): The argument parser instance.

    Returns:
        dict[str, Any]: A dictionary of parsed arguments.
    """
    parser.add_argument("--env", type=str, required=True, help="Environment")

    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Location of training data")
    parser.add_argument("--val_data", type=str, required=True, help="Location of validation data")

    parser.add_argument(
        "--num_train_trajs", type=int, default=100, help="Number of random training trajectories to visualize"
    )
    parser.add_argument(
        "--num_train_steps", type=int, default=100, help="Number of random training steps to visualize per trajectory"
    )
    parser.add_argument(
        "--num_val_trajs", type=int, default=100, help="Number of random validation trajectories to visualize"
    )
    parser.add_argument(
        "--num_val_steps", type=int, default=100, help="Number of random validation steps to visualize per trajectory"
    )
    parser.add_argument("--save_imgs", type=str, default="sample_images", help="Directory to which to save images")

    # parse arguments
    args = parser.parse_args()
    args_dict: dict[str, Any] = vars(args)
    print_args(args)

    train_file_name = os.path.splitext(os.path.basename(args_dict["train_data"]))[0]
    val_file_name = os.path.splitext(os.path.basename(args_dict["val_data"]))[0]
    # make save directory
    train_imgs_save_dir: str = f"{args_dict['save_imgs']}/{train_file_name}/train"
    val_imgs_save_dir: str = f"{args_dict['save_imgs']}/{val_file_name}/val"
    args_dict["train_imgs_save_dir"] = train_imgs_save_dir
    args_dict["val_imgs_save_dir"] = val_imgs_save_dir
    os.makedirs(train_imgs_save_dir, exist_ok=True)
    os.makedirs(val_imgs_save_dir, exist_ok=True)

    return args_dict


def run_with_argsd(args_dict: dict[str, Any]) -> None:
    """Run visualization given a parsed args dict (shared by CLI/programmatic)."""
    print("Loading data ...")
    start_time = time.time()
    state_episodes_train: list[NDArray[float32]]
    state_episodes_val: list[NDArray[float32]]

    with open(args_dict["train_data"], "rb") as train_file, open(args_dict["val_data"], "rb") as val_file:
        state_episodes_train, action_episodes_train = pickle.load(train_file)
        state_episodes_val, action_episodes_val = pickle.load(val_file)

    print(f"{len(state_episodes_train)} train episodes, {len(state_episodes_val)} val episodes")
    print(f"Data load time: {time.time() - start_time}")

    train_states_np: NDArray[float32] = np.stack(list(state_episodes_train), axis=0)
    train_actions_np: NDArray[intp] = np.stack(list(action_episodes_train), axis=0).astype(intp)
    num_train_states: int = train_states_np.shape[0]
    assert (args_dict["num_train_trajs"] <= num_train_states) and (args_dict["num_train_trajs"] >= 0)
    num_train_episode_steps = train_states_np.shape[1]
    assert (args_dict["num_train_steps"] <= num_train_episode_steps) and (args_dict["num_train_steps"] >= 0)

    val_states_np: NDArray[float32] = np.stack(list(state_episodes_val), axis=0)
    val_actions_np: NDArray[intp] = np.stack(list(action_episodes_val), axis=0).astype(intp)
    num_val_states: int = val_states_np.shape[0]
    assert (args_dict["num_val_trajs"] <= num_val_states) and (args_dict["num_val_trajs"] >= 0)
    num_val_episode_steps = val_states_np.shape[1]
    assert (args_dict["num_val_steps"] <= num_val_episode_steps) and (args_dict["num_val_steps"] >= 0)

    unique_train_episodes_idxs = np.random.choice(
        np.arange(num_train_states), size=args_dict["num_train_trajs"], replace=False
    )
    unique_train_steps_idxs = np.array([
        np.random.choice(np.arange(num_train_episode_steps), size=args_dict["num_train_steps"], replace=False)
        for _ in range(args_dict["num_train_trajs"])
    ])
    unique_train_steps_idxs.sort(axis=1)

    unique_val_episodes_idxs = np.random.choice(
        np.arange(num_val_states), size=args_dict["num_val_trajs"], replace=False
    )
    unique_val_steps_idxs = np.array([
        np.random.choice(np.arange(num_val_episode_steps), size=args_dict["num_val_steps"], replace=False)
        for _ in range(args_dict["num_val_trajs"])
    ])
    unique_val_steps_idxs.sort(axis=1)

    unique_train_states = train_states_np[unique_train_episodes_idxs[:, None], unique_train_steps_idxs]
    unique_val_states = val_states_np[unique_val_episodes_idxs[:, None], unique_val_steps_idxs]

    # unique_train_states_actions = train_states_actions_np[unique_train_episodes_idxs[:, None],
    #   unique_train_steps_idxs]
    # unique_val_states_actions = unique_val_states_actions_np[unique_val_episodes_idxs[:, None],
    #   unique_val_steps_idxs]

    plot_images(
        unique_train_states,
        args_dict["train_imgs_save_dir"],
        train_actions_np,
        unique_train_episodes_idxs,
        unique_train_steps_idxs,
        args_dict["env"],
    )
    plot_images(
        unique_val_states,
        args_dict["val_imgs_save_dir"],
        val_actions_np,
        unique_val_episodes_idxs,
        unique_val_steps_idxs,
        args_dict["env"],
    )

    print(f"\nSample images from the data have been saved to:\n\t{args_dict['save_imgs']}")


def run_offline_data_viz(cfg: OfflineDataVizConfig) -> None:
    """Programmatic entry using a typed config."""
    args_dict = {
        "env": cfg.env,
        "train_data": cfg.train_data,
        "val_data": cfg.val_data,
        "num_train_trajs": cfg.num_train_trajs,
        "num_train_steps": cfg.num_train_steps,
        "num_val_trajs": cfg.num_val_trajs,
        "num_val_steps": cfg.num_val_steps,
        "save_imgs": cfg.save_imgs,
    }

    train_file_name = os.path.splitext(os.path.basename(str(args_dict["train_data"])))[0]
    val_file_name = os.path.splitext(os.path.basename(str(args_dict["val_data"])))[0]
    train_imgs_save_dir: str = f"{args_dict['save_imgs']}/{train_file_name}/train"
    val_imgs_save_dir: str = f"{args_dict['save_imgs']}/{val_file_name}/val"
    args_dict["train_imgs_save_dir"] = train_imgs_save_dir
    args_dict["val_imgs_save_dir"] = val_imgs_save_dir
    os.makedirs(train_imgs_save_dir, exist_ok=True)
    os.makedirs(val_imgs_save_dir, exist_ok=True)
    run_with_argsd(args_dict)


def main() -> None:
    """Main function to load data, process it, and plot images."""
    parser: ArgumentParser = ArgumentParser()
    args_dict: dict[str, Any] = parse_arguments(parser)
    run_with_argsd(args_dict)


if __name__ == "__main__":
    main()
