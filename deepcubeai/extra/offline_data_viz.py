import os
import pickle
import time
from argparse import ArgumentParser
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deepcubeai.utils.data_utils import print_args


def save_image(image_np: np.ndarray, save_path: str) -> None:
    """Saves a NumPy array as an image file.

    Args:
        image_np (np.ndarray): The image data as a NumPy array of shape (C, H, W).
        save_path (str): The path where the image will be saved.
    """
    image_np = image_np * 255
    image_np = image_np.astype(np.uint8)

    # Transpose the array to match the RGB format
    image_np_t = np.transpose(image_np, (1, 2, 0))
    image = Image.fromarray(image_np_t)
    image.save(save_path)


def plot_images(
    images_np: np.ndarray,
    save_dir: str,
    action_episodes: np.ndarray,
    episode_idxs: np.ndarray,
    step_idxs: np.ndarray,
    env: str,
) -> None:
    """Plots and saves images from episodes and steps.

    Args:
        images_np (np.ndarray): The image data as a NumPy array.
        save_dir (str): The directory where the images will be saved.
        action_episodes (np.ndarray): The actions taken in each episode.
        episode_idxs (np.ndarray): The indices of the episodes.
        step_idxs (np.ndarray): The indices of the steps within each episode.
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
                save_image(
                    images_np[i, j],
                    f"{save_dir}/{k}_{env}_seq{episode_idxs[i]}_step{step_idxs[i, j]}.png",
                )
                save_image(
                    images_np[i, j + 1],
                    f"{save_dir}/{k}_{env}_seq{episode_idxs[i]}_step{step_idxs[i, j+1]}.png",
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
        _, axes = plt.subplots(1, images_np.shape[1], figsize=(images_np.shape[1] * 2 * ratio, 2))

        for j in range(images_np.shape[1]):
            if step_idxs[i, j] == 0:
                action_taken = "None"
            else:
                action_taken = str(action_episodes[episode_idxs[i], step_idxs[i, j] - 1])

            axes[j].imshow(images_np[i, j].transpose(1, 2, 0))
            # axes[j].set_xticks(np.arange(0, images_np.shape[-1], 10))
            # axes[j].set_yticks(np.arange(0, images_np.shape[-2], 10))
            axes[j].tick_params(axis="both", which="both", labelsize=6)
            axes[j].set_title(
                f"Episode {episode_idxs[i]}, Step {step_idxs[i, j]}\nAction Taken: {action_taken}",
                size=8,
                pad=5,
            )

        plt.savefig(f"{save_dir}/e{episode_idxs[i]}.png", dpi=300)
        # plt.show()
        plt.close()


def parse_arguments(parser: ArgumentParser) -> Dict[str, Any]:
    """Parses command-line arguments.

    Args:
        parser (ArgumentParser): The argument parser instance.

    Returns:
        Dict[str, Any]: A dictionary of parsed arguments.
    """
    parser.add_argument("--env", type=str, required=True, help="Environment")

    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Location of training data")
    parser.add_argument("--val_data", type=str, required=True, help="Location of validation data")

    parser.add_argument("--num_train_trajs",
                        type=int,
                        default=100,
                        help="Number of random training trajectories to visualize")
    parser.add_argument("--num_train_steps",
                        type=int,
                        default=100,
                        help="Number of random training steps to visualize per trajectory")
    parser.add_argument("--num_val_trajs",
                        type=int,
                        default=100,
                        help="Number of random validation trajectories to visualize")
    parser.add_argument("--num_val_steps",
                        type=int,
                        default=100,
                        help="Number of random validation steps to visualize per trajectory")
    parser.add_argument("--save_imgs",
                        type=str,
                        default="sample_images",
                        help="Directory to which to save images")

    # parse arguments
    args = parser.parse_args()
    args_dict: Dict[str, Any] = vars(args)
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


def main() -> None:
    """Main function to load data, process it, and plot images."""
    parser: ArgumentParser = ArgumentParser()
    args_dict: Dict[str, Any] = parse_arguments(parser)

    print("Loading data ...")
    start_time = time.time()
    state_episodes_train: List[np.ndarray]
    state_episodes_val: List[np.ndarray]

    with open(args_dict["train_data"], "rb") as train_file, open(args_dict["val_data"],
                                                                 "rb") as val_file:
        state_episodes_train, action_episodes_train = pickle.load(train_file)
        state_episodes_val, action_episodes_val = pickle.load(val_file)

    print(f"{len(state_episodes_train)} train episodes, {len(state_episodes_val)} val episodes")
    print(f"Data load time: {time.time() - start_time}")

    train_states_np: np.ndarray = np.stack(
        [state_episode_train for state_episode_train in state_episodes_train], axis=0)
    train_actions_np: np.ndarray = np.stack(
        [action_episodes_train for action_episodes_train in action_episodes_train], axis=0)
    num_train_states: int = train_states_np.shape[0]
    assert (args_dict["num_train_trajs"] <= num_train_states) and (args_dict["num_train_trajs"]
                                                                   >= 0)
    num_train_episode_steps = train_states_np.shape[1]
    assert (args_dict["num_train_steps"]
            <= num_train_episode_steps) and (args_dict["num_train_steps"] >= 0)

    val_states_np: np.ndarray = np.stack(
        [state_episode_val for state_episode_val in state_episodes_val], axis=0)
    val_actions_np: np.ndarray = np.stack(
        [action_episodes_val for action_episodes_val in action_episodes_val], axis=0)
    num_val_states: int = val_states_np.shape[0]
    assert (args_dict["num_val_trajs"] <= num_val_states) and (args_dict["num_val_trajs"] >= 0)
    num_val_episode_steps = val_states_np.shape[1]
    assert (args_dict["num_val_steps"] <= num_val_episode_steps) and (args_dict["num_val_steps"]
                                                                      >= 0)

    unique_train_episodes_idxs = np.random.choice(np.arange(num_train_states),
                                                  size=args_dict["num_train_trajs"],
                                                  replace=False)
    unique_train_steps_idxs = np.array([
        np.random.choice(
            np.arange(num_train_episode_steps),
            size=args_dict["num_train_steps"],
            replace=False,
        ) for _ in range(args_dict["num_train_trajs"])
    ])
    unique_train_steps_idxs.sort(axis=1)

    unique_val_episodes_idxs = np.random.choice(np.arange(num_val_states),
                                                size=args_dict["num_val_trajs"],
                                                replace=False)
    unique_val_steps_idxs = np.array([
        np.random.choice(np.arange(num_val_episode_steps),
                         size=args_dict["num_val_steps"],
                         replace=False) for _ in range(args_dict["num_val_trajs"])
    ])
    unique_val_steps_idxs.sort(axis=1)

    unique_train_states = train_states_np[unique_train_episodes_idxs[:, None],
                                          unique_train_steps_idxs]
    unique_val_states = val_states_np[unique_val_episodes_idxs[:, None], unique_val_steps_idxs]

    # unique_train_states_actions = train_states_actions_np[unique_train_episodes_idxs[:, None],
    #   unique_train_steps_idxs]
    # unique_val_states_actions = unique_val_states_actions_np[unique_val_episodes_idxs[:, None],
    #   unique_val_steps_idxs]

    plot_images(unique_train_states, args_dict["train_imgs_save_dir"], train_actions_np,
                unique_train_episodes_idxs, unique_train_steps_idxs, args_dict["env"])
    plot_images(unique_val_states, args_dict["val_imgs_save_dir"], val_actions_np,
                unique_val_episodes_idxs, unique_val_steps_idxs, args_dict["env"])

    print("\nAll of the sample images are saved to file.")


if __name__ == "__main__":
    main()
