import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from torch import nn

from deepcubeai.environments.environment_abstract import Environment, State


class ImageHandler:

    def __init__(self, env: Environment, state_soln: State, decoder: nn.Module,
                 path: List[np.array], len_soln: int, state_idx: int, device: torch.device):
        """Initializes the ImageHandler.

        Args:
            env (Environment): The environment instance.
            state_soln (State): The solution state.
            decoder (nn.Module): The decoder neural network.
            path (List[np.array]): The path of states.
            len_soln (int): The length of the solution.
            state_idx (int): The index of the state.
            device (torch.device): The device to run computations on.
        """
        self.img_texts: List[List[str]] = []
        self.images_dict: Dict[str, np.ndarray] = {}
        self.img_texts_row1: List[str] = []
        self.img_texts_row2: List[str] = []
        self.img_texts_row3: List[str] = []
        self.env: Environment = env
        self.state_soln: State = state_soln
        self.decoder: nn.Module = decoder
        self.path: List[np.array] = path
        self.len_soln: int = len_soln
        self.state_idx: int = state_idx
        self.device: torch.device = device
        self.has_get_solution: bool = bool(hasattr(self.state_soln, "get_solution"))

        # Initialize for the first image (step 0)
        self.state_images_np: np.ndarray = np.array(self.env.state_to_real([self.state_soln])[0])
        self.nnet_images_np: np.ndarray = np.array(
            self._get_real_state_image([self.path[0]], self.device))

        if self.state_images_np.shape[0] == 6:
            self.state_images_np = np.concatenate(
                (self.state_images_np[:3, :, :], self.state_images_np[3:, :, :]), axis=2)
            self.nnet_images_np = np.concatenate(
                (self.nnet_images_np[:3, :, :], self.nnet_images_np[3:, :, :]), axis=2)

        self.state_images_np = np.expand_dims(self.state_images_np, axis=0)
        self.nnet_images_np = np.expand_dims(self.nnet_images_np, axis=0)
        self.img_texts_row1.append("Step 0 - Move: None")
        self.img_texts_row2.append("Step 0 - Move: None")

        self.optimal_path_len: int = 0
        if hasattr(self.state_soln, "get_solution"):
            self.has_get_solution = True
            self.state_soln_optimal: State = self.state_soln
            self.optimal_soln: List[int] = self.state_soln_optimal.get_solution()
            self.optimal_path_len = len(self.optimal_soln)
            self.state_images_optimal_np: np.ndarray = np.array(
                self.env.state_to_real([self.state_soln_optimal])[0])
            self.state_images_optimal_np = np.expand_dims(self.state_images_optimal_np, axis=0)
            self.img_texts_row3.append("Step 0 - Move: None")

    def step(self, idx: int, state_soln: State, move: int) -> None:
        """Updates the state and images for a given step.

        Args:
            idx (int): The index of the step.
            state_soln (State): The solution state.
            move (int): The move taken.
        """
        self.state_soln = state_soln
        state_img_np: np.ndarray = self.env.state_to_real([self.state_soln])[0]
        nnet_img_np: np.ndarray = self._get_real_state_image([self.path[idx + 1]], self.device)
        if self.has_get_solution and idx < self.optimal_path_len:
            self.state_soln_optimal = self.env.next_state([self.state_soln_optimal],
                                                          [self.optimal_soln[idx]])[0][0]
            state_img_optimal_np = self.env.state_to_real([self.state_soln_optimal])[0]

        if state_img_np.shape[0] == 6:
            state_img_np = np.concatenate((state_img_np[:3, :, :], state_img_np[3:, :, :]), axis=2)
            nnet_img_np = np.concatenate((nnet_img_np[:3, :, :], nnet_img_np[3:, :, :]), axis=2)
            if self.has_get_solution and idx < self.optimal_path_len:
                state_img_optimal_np = np.concatenate(
                    (state_img_optimal_np[:3, :, :], state_img_optimal_np[3:, :, :]), axis=2)

        self.state_images_np = np.vstack((self.state_images_np, np.expand_dims(state_img_np,
                                                                               axis=0)))
        self.nnet_images_np = np.vstack((self.nnet_images_np, np.expand_dims(nnet_img_np, axis=0)))
        if self.has_get_solution and idx < self.optimal_path_len:
            self.state_images_optimal_np = np.vstack(
                (self.state_images_optimal_np, np.expand_dims(state_img_optimal_np, axis=0)))

        self.img_texts_row1.append(f"Step {idx + 1} - Move: {move}")
        self.img_texts_row2.append(f"Step {idx + 1} - Move: {move}")
        if self.has_get_solution and idx < self.optimal_path_len:
            self.img_texts_row3.append(f"Step {idx + 1} - Move: {self.optimal_soln[idx]}")

    def save_images(self, save_imgs_dir: str) -> None:
        """Saves the images to the specified directory.

        Args:
            save_imgs_dir (str): The directory to save images.
        """
        self._prepare_images_to_save()
        self._save_as_img(self.images_dict, self.img_texts, save_imgs_dir, self.state_idx)

    def _prepare_images_to_save(self) -> None:
        """Prepares the images for saving."""
        key_real_world: str = "The Path Found: Actions Taken in the Real Environment"
        key_recon: str = ("The Path Found: Actions Taken in the NNet Environment Model " +
                          "(Reconstructions)")
        key_optimal: str = "The Optimal Path: Actions Taken in the Real Environment"

        self.state_images_np = self.state_images_np.transpose(0, 2, 3, 1)
        self.nnet_images_np = self.nnet_images_np.transpose(0, 2, 3, 1)

        self.images_dict[key_real_world] = self.state_images_np[:]
        self.images_dict[key_recon] = self.nnet_images_np[:]
        self.img_texts.append(self.img_texts_row1)
        self.img_texts.append(self.img_texts_row2)

        if self.has_get_solution:
            # Add the remaining if there is any
            for idx in range(self.len_soln, self.optimal_path_len):
                self.state_soln_optimal = self.env.next_state([self.state_soln_optimal],
                                                              [self.optimal_soln[idx]])[0][0]
                state_img_optimal_np = self.env.state_to_real([self.state_soln_optimal])[0]

                self.state_images_optimal_np = np.vstack(
                    (self.state_images_optimal_np, np.expand_dims(state_img_optimal_np, axis=0)))
                self.img_texts_row3.append(f"Step {idx} - Move: {self.optimal_soln[idx]}")

            self.images_dict[key_optimal] = self.state_images_optimal_np.transpose(0, 2, 3, 1)
            self.img_texts.append(self.img_texts_row3)

    def _get_real_state_image(self, state_enc: List[np.array], device: torch.device) -> np.ndarray:
        """Gets the real state image from the encoded state.

        Args:
            state_enc (List[np.array]): The encoded state.
            device (torch.device): The device to run computations on.

        Returns:
            np.ndarray: The real state image.
        """
        state_dec = self.decoder(torch.tensor(np.array(state_enc), device=device).float().detach())
        image_np = np.clip(state_dec[0].detach().cpu().data.numpy(), 0, 1)
        return image_np

    def _calculate_figure_parameters(self, w: int, h: int) -> Tuple[float, float, float, int]:
        """Calculates the figure parameters for plotting.

        Args:
            w (int): The width of the image.
            h (int): The height of the image.

        Returns:
            Tuple[float, float, float, int]: The width ratio, height ratio, middle ratio, and
                padding.
        """
        if w > h:
            w_ratio = (w / h) * 0.75
            h_ratio = 1.0
            middle_ratio = 0.2
            pad = 20
        elif w < h:
            h_ratio = (h / w) * 0.75
            w_ratio = 1.0
            middle_ratio = 0.3
            pad = 30
        else:
            h_ratio = 1.0
            w_ratio = 1.0
            middle_ratio = 0.2
            pad = 25
        return w_ratio, h_ratio, middle_ratio, pad

    def _create_figure(self,
                       num_cols: int,
                       num_rows: int,
                       w_ratio: float,
                       h_ratio: float,
                       middle_ratio: float,
                       dpi: int = 150) -> Tuple[plt.Figure, gridspec.GridSpec]:
        """Creates a figure for plotting images.

        Args:
            num_cols (int): The number of columns.
            num_rows (int): The number of rows.
            w_ratio (float): The width ratio.
            h_ratio (float): The height ratio.
            middle_ratio (float): The middle ratio.
            dpi (int, optional): The dots per inch for the figure. Defaults to 150.

        Returns:
            Tuple[plt.Figure, gridspec.GridSpec]: The figure and grid specification.
        """
        height_ratios_blanks = [1, middle_ratio] * num_rows
        height_ratios_blanks = height_ratios_blanks[:-1]

        fig = plt.figure(figsize=(4 * num_cols * w_ratio, 4 * num_rows * h_ratio), dpi=dpi)
        gs = gridspec.GridSpec(2 * num_rows - 1, num_cols, height_ratios=height_ratios_blanks)

        return fig, gs

    def _save_as_img(self, images_dict: Dict[str, np.ndarray], img_texts: List[List[str]],
                     save_imgs_dir: str, state_idx: int) -> None:
        """Saves the images as a file.

        Args:
            images_dict (Dict[str, np.ndarray]): The dictionary of images.
            img_texts (List[List[str]]): The list of image texts.
            save_imgs_dir (str): The directory to save images.
            state_idx (int): The index of the state.
        """
        h = list(images_dict.values())[0].shape[1]
        w = list(images_dict.values())[0].shape[2]

        h_ratio, w_ratio, middle_ratio, pad = self._calculate_figure_parameters(w, h)
        keys_lst: List[str] = list(images_dict.keys())

        max_length = 0
        for array in list(images_dict.values()):
            length = array.shape[0]
            max_length = max(max_length, length)

        num_cols = max_length
        num_rows = len(keys_lst)

        fig, gs = self._create_figure(num_cols, num_rows, w_ratio, h_ratio, middle_ratio)

        for row in range(0, 2 * num_rows - 1, 2):
            image_dict_row = images_dict[keys_lst[row // 2]]
            for col in range(len(image_dict_row)):
                ax = fig.add_subplot(gs[row, col])
                ax.set_title(img_texts[row // 2][col])
                ax.tick_params(axis="both", which="both", labelsize="small")
                ax.imshow(image_dict_row[col])

                # Ghost axes and titles on gs
                ax_ghost = fig.add_subplot(gs[row, :])
                ax_ghost.axis("off")
                ax_ghost.set_title(keys_lst[row // 2], pad=pad, fontweight="bold")

        save_file_path = os.path.join(save_imgs_dir, f"state_{state_idx}.png")

        fig.savefig(save_file_path, bbox_inches="tight", pad_inches=0.7)
        # plt.show()
        plt.close()


def is_valid_soln(state: State,
                  state_goal: State,
                  soln: List[int],
                  env: Environment,
                  decoder: Optional[nn.Module] = None,
                  device: Optional[torch.device] = None,
                  state_idx: Optional[int] = None,
                  path: Optional[List[np.array]] = None,
                  save_imgs_dir: Optional[str] = None,
                  save_imgs: bool = False) -> bool:
    """Checks if the solution is valid.

    Args:
        state (State): The initial state.
        state_goal (State): The goal state.
        soln (List[int]): The list of moves.
        env (Environment): The environment instance.
        decoder (Optional[nn.Module], optional): The decoder neural network. Defaults to None.
        device (Optional[torch.device], optional): The device to run computations on. Defaults to
            None.
        state_idx (Optional[int], optional): The index of the state. Defaults to None.
        path (Optional[List[np.array]], optional): The path of states. Defaults to None.
        save_imgs_dir (Optional[str], optional): The directory to save images. Defaults to None.
        save_imgs (bool, optional): Whether to save images. Defaults to False.

    Returns:
        bool: True if the solution is valid, False otherwise.
    """
    state_soln: State = state
    move: int

    if save_imgs:
        assert all(
            param is not None for param in [decoder, device, state_idx, path, save_imgs_dir]
        ), ("decoder, device, state_idx, path, and save_imgs_dir must be provided when save_imgs "
            + "is True")
        image_handler = ImageHandler(env, state_soln, decoder, path, len(soln), state_idx, device)

    for idx, move in enumerate(soln):
        state_soln = env.next_state([state_soln], [move])[0][0]
        if save_imgs:
            image_handler.step(idx, state_soln, move)

    if save_imgs:
        image_handler.save_images(save_imgs_dir)

    return env.is_solved([state_soln], [state_goal])[0]
