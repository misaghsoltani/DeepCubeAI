import os
import pickle
import zipfile
from typing import Any, List, Tuple

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from deepcubeai.environments.environment_abstract import Environment, State
from deepcubeai.utils.pytorch_models import (
    Conv2dModel,
    FullyConnectedModel,
    ResnetModel,
    STEThresh,
)


class SokobanDQN(nn.Module):

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, input_dim: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        """
        Initializes the SokobanDQN model.

        Args:
            input_dim (int): Input dimension.
            h1_dim (int): Dimension of the first hidden layer.
            resnet_dim (int): Dimension of the ResNet blocks.
            num_resnet_blocks (int): Number of ResNet blocks.
            out_dim (int): Output dimension.
            batch_norm (bool): Whether to use batch normalization.
        """
        super().__init__()
        self.first_fc = FullyConnectedModel(input_dim * 2, [h1_dim, resnet_dim], [batch_norm] * 2,
                                            ["RELU"] * 2)
        self.resnet = ResnetModel(resnet_dim, num_resnet_blocks, out_dim, batch_norm, "RELU")

    def forward(self, states: Tensor, states_goal: Tensor) -> Tensor:
        """
        Forward pass for the SokobanDQN model.

        Args:
            states (Tensor): Input tensor representing the states.
            states_goal (Tensor): Input tensor representing the goal states.

        Returns:
            Tensor: Output tensor after passing through the model.
        """
        x = self.first_fc(torch.cat((states, states_goal), dim=1))
        x = self.resnet(x)

        return x


class Encoder(nn.Module):

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, chan_in: int, chan_enc: int):
        """
        Initializes the Encoder model.

        Args:
            chan_in (int): Number of input channels.
            chan_enc (int): Number of encoded channels.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            Conv2dModel(chan_in, [16, chan_enc], [2, 2], [0, 0], [True, False],
                        ["RELU", "SIGMOID"],
                        strides=[2, 2]),
            nn.Flatten(),
        )

        self.ste_thresh = STEThresh()

    def forward(self, states: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the Encoder model.

        Args:
            states (Tensor): Input tensor representing the states.

        Returns:
            Tuple[Tensor, Tensor]: Encoded states and thresholded encoded states.
        """
        encs = self.encoder(states)
        encs_d = self.ste_thresh.apply(encs, 0.5)

        return encs, encs_d


class Decoder(nn.Module):

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, chan_in: int, chan_enc: int):
        """
        Initializes the Decoder model.

        Args:
            chan_in (int): Number of input channels.
            chan_enc (int): Number of encoded channels.
        """
        super().__init__()

        self.chan_enc: int = chan_enc

        self.decoder_conv = nn.Sequential(
            Conv2dModel(self.chan_enc, [16, 16], [2, 2], [0, 0], [True, False],
                        ["RELU", "SIGMOID"],
                        strides=[2, 2],
                        transpose=True),
            Conv2dModel(16, [chan_in], [1], [0], [False], ["LINEAR"]),
        )

    def forward(self, encs: Tensor) -> Tensor:
        """
        Forward pass for the Decoder model.

        Args:
            encs (Tensor): Input tensor representing the encoded states.

        Returns:
            Tensor: Decoded states.
        """
        decs = torch.reshape(encs, (encs.shape[0], self.chan_enc, 10, 10))
        decs = self.decoder_conv(decs)

        return decs


class EnvModel(nn.Module):

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, num_actions: int, chan_enc: int):
        """
        Initializes the EnvModel.

        Args:
            num_actions (int): Number of actions.
            chan_enc (int): Number of encoded channels.
        """
        super().__init__()
        self.num_actions = num_actions

        self.chan_enc: int = chan_enc
        self.mask_net = nn.Sequential(
            Conv2dModel(self.chan_enc + 4, [32, 32, self.chan_enc], [3, 3, 3], [1, 1, 1],
                        [True, True, False], ["RELU", "RELU", "SIGMOID"]),
            nn.Flatten(),
        )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """
        Forward pass for the EnvModel.

        Args:
            states (Tensor): Input tensor representing the states.
            actions (Tensor): Input tensor representing the actions.

        Returns:
            Tensor: Next states after applying the actions.
        """
        states_conv = states.view(-1, self.chan_enc, 10, 10)

        actions_oh = F.one_hot(actions.long(), self.num_actions)
        actions_oh = actions_oh.float()

        actions_oh = actions_oh.view(-1, self.num_actions, 1, 1)
        actions_oh = actions_oh.repeat(1, 1, states_conv.shape[2], states_conv.shape[3])

        states_actions = torch.cat((states_conv.float(), actions_oh), dim=1)
        states_next = self.mask_net(states_actions)

        return states_next


class EnvModelContinuous(nn.Module):

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, num_actions: int, chan_in: int, chan_enc: int):
        """
        Initializes the EnvModelContinuous.

        Args:
            num_actions (int): Number of actions.
            chan_in (int): Number of input channels.
            chan_enc (int): Number of encoded channels.
        """
        super().__init__()
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            Conv2dModel(chan_in, [16, chan_enc], [2, 2], [0, 0], [True, False],
                        ["RELU", "SIGMOID"],
                        strides=[2, 2]))

        self.env_model = nn.Sequential(
            Conv2dModel(chan_enc + 4, [32, 32, chan_enc], [3, 3, 3], [1, 1, 1],
                        [True, True, False], ["RELU", "RELU", "RELU"]),
            Conv2dModel(chan_enc, [16, 16], [2, 2], [0, 0], [True, False], ["RELU", "SIGMOID"],
                        strides=[2, 2],
                        transpose=True), Conv2dModel(16, [chan_in], [1], [0], [False], ["LINEAR"]))

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """
        Forward pass for the EnvModelContinuous.

        Args:
            states (Tensor): Input tensor representing the states.
            actions (Tensor): Input tensor representing the actions.

        Returns:
            Tensor: Next states after applying the actions.
        """
        # encode
        states_conv = self.encoder(states)

        # preprocess actions
        actions_oh = F.one_hot(actions.long(), self.num_actions)
        actions_oh = actions_oh.float()

        actions_oh = actions_oh.view(-1, self.num_actions, 1, 1)
        actions_oh = actions_oh.repeat(1, 1, states_conv.shape[2], states_conv.shape[3])

        # get next states
        states_actions = torch.cat((states_conv.float(), actions_oh), dim=1)
        states_next = self.env_model(states_actions)

        return states_next


class SokobanState(State):
    __slots__ = ["agent", "walls", "boxes", "hash", "seed"]

    def __init__(self, agent: np.array, boxes: np.ndarray, walls: np.ndarray, seed: int = None):
        """
        Initializes the SokobanState.

        Args:
            agent (np.array): Agent's position.
            boxes (np.ndarray): Boxes' positions.
            walls (np.ndarray): Walls' positions.
            seed (int, optional): Random seed. Defaults to None.
        """
        super().__init__()
        self.agent: np.array = agent
        self.boxes: np.ndarray = boxes
        self.walls: np.ndarray = walls
        self.seed = seed

        self.hash = None

    def __hash__(self) -> int:
        """
        Computes the hash of the state.

        Returns:
            int: Hash value of the state.
        """
        if self.hash is None:
            boxes_flat = self.boxes.flatten()
            walls_flat = self.walls.flatten()
            state = np.concatenate((self.agent, boxes_flat, walls_flat), axis=0)

            self.hash = hash(state.tobytes())

        return self.hash

    def __eq__(self, other: Any) -> bool:
        """
        Checks if two states are equal.

        Args:
            other (Any): Another state to compare with.

        Returns:
            bool: True if the states are equal, False otherwise.
        """
        agents_eq: bool = np.array_equal(self.agent, other.agent)
        boxes_eq: bool = np.array_equal(self.boxes, other.boxes)
        walls_eq: bool = np.array_equal(self.walls, other.walls)

        return agents_eq and boxes_eq and walls_eq


def load_states(file_name: str) -> List[SokobanState]:
    """
    Loads Sokoban states from a file.

    Args:
        file_name (str): Path to the file containing the states.

    Returns:
        List[SokobanState]: List of loaded Sokoban states.
    """
    states_np = pickle.load(open(file_name, "rb"))
    states: List[SokobanState] = []

    agent_idxs = np.where(states_np == 1)
    box_masks = states_np == 2
    wall_masks = states_np == 4

    for idx in range(states_np.shape[0]):
        agent_idx = np.array([agent_idxs[1][idx], agent_idxs[2][idx]], dtype=int)

        states.append(SokobanState(agent_idx, box_masks[idx], wall_masks[idx]))

    return states


def _get_surfaces() -> List[np.ndarray]:
    """
    Loads surface images for Sokoban.

    Returns:
        List[np.ndarray]: List of surface images.
    """
    img_dir = "deepcubeai/environments/sokoban_data/surface"

    # Load images, representing the corresponding situation
    box = imageio.imread(f"{img_dir}/box.png")
    floor = imageio.imread(f"{img_dir}/floor.png")
    player = imageio.imread(f"{img_dir}/player.png")
    wall = imageio.imread(f"{img_dir}/wall.png")

    surfaces = [wall, floor, player, box]

    return surfaces


def _env_data_exists(dir: str, item_name: str) -> bool:
    """
    Checks if the specified item (file or folder) exists and processes a ZIP file if needed.

    This function performs the following actions:
    1. Checks if the `item_name` (file or folder) exists in the given directory (`dir`).
    2. If the `item_name` is not a ZIP file and exists, returns True.
    3. If a corresponding ZIP file (`item_name` with a `.zip` extension) exists,
       extracts its contents to the given directory and returns True.
    4. If neither the `item_name` nor the ZIP file is found, prints a message and returns False.

    Args:
        dir (str): The directory to search in.
        item_name (str): The name of the file or folder to check for.

    Returns:
        bool: True if the item or its corresponding ZIP file exists and was processed,
              False otherwise.
    """
    name, ext = os.path.splitext(item_name)
    name_zip: str = f"{name}.zip"
    item_path: str = os.path.join(dir, item_name)
    zip_path: str = os.path.join(dir, name_zip)

    if ext != ".zip" and os.path.exists(item_path):
        return True

    if os.path.exists(zip_path):
        print(f"ZIP file '{name_zip}' found in '{dir}'.\nExtracting contents...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extracted_files: List[str] = zip_ref.namelist()
            if not extracted_files:
                raise ValueError(
                    f"The ZIP file '{name_zip}' is empty or does not contain any files.")

            zip_ref.extractall(dir)

        print(f"Successfully extracted all contents from '{name_zip}' to '{dir}'.")
        return True

    print(f"Neither '{item_name}' nor '{name_zip}' found in '{dir}'. One of these is required.")
    return False


class Sokoban(Environment):

    def __init__(self, dim: int = 10, num_boxes: int = 4):
        """
        Initializes the Sokoban environment.

        Args:
            dim (int): Dimension of the environment. Default is 10.
            num_boxes (int): Number of boxes in the environment. Default is 3.
        """
        super().__init__()

        self.dim: int = dim
        self.num_boxes: int = num_boxes

        self.num_moves: int = 4

        goal_states_dir: str = "deepcubeai/environments/sokoban_data"
        goal_states_filename: str = "goal_states.pkl"
        assert _env_data_exists(goal_states_dir, goal_states_filename), (
            f"Expected '{goal_states_filename}' or '{goal_states_filename[:-4]}.zip' to " +
            f"exist in '{goal_states_dir}'")

        self.states_train: List[SokobanState] = load_states(
            os.path.join(goal_states_dir, goal_states_filename))

        self.img_dim: int = 40
        self.chan_enc: int = 16
        enc_h: int = 10
        enc_w: int = 10
        self.enc_dim: int = enc_h * enc_w * self.chan_enc

        self.enc_hw: Tuple[int, int] = (enc_h, enc_w)

        self._surfaces = _get_surfaces()

    def get_env_name(self) -> str:
        """Gets the name of the environment.

        Returns:
            str: The name of the environment, "sokoban".
        """
        return "sokoban"

    @property
    def num_actions_max(self) -> int:
        """
        Returns the maximum number of actions.

        Returns:
            int: Maximum number of actions.
        """
        return self.num_moves

    def rand_action(self, states: List[State]) -> List[int]:
        """
        Generates random actions for the given states.

        Args:
            states (List[State]): List of states.

        Returns:
            List[int]: List of random actions.
        """
        return list(np.random.randint(0, self.num_moves, size=len(states)))

    def next_state(self, states: List[SokobanState],
                   actions: List[int]) -> Tuple[List[SokobanState], List[float]]:
        """
        Computes the next state and transition cost given the current state and action.

        Args:
            states (List[SokobanState]): List of current states.
            actions (List[int]): List of actions to take.

        Returns:
            Tuple[List[SokobanState], List[float]]: Next states and transition costs.
        """
        agent = np.stack([state.agent for state in states], axis=0)
        boxes = np.stack([state.boxes for state in states], axis=0)
        walls_next = np.stack([state.walls for state in states], axis=0)

        idxs_arange = np.arange(0, len(states))
        agent_next_tmp = self._get_next_idx(agent, actions)
        agent_next = np.zeros(agent_next_tmp.shape, dtype=int)

        boxes_next = boxes.copy()

        # agent -> wall
        agent_wall = walls_next[idxs_arange, agent_next_tmp[:, 0], agent_next_tmp[:, 1]]
        agent_next[agent_wall] = agent[agent_wall]

        # agent -> box
        agent_box = boxes[idxs_arange, agent_next_tmp[:, 0], agent_next_tmp[:, 1]]
        boxes_next_tmp = self._get_next_idx(agent_next_tmp, actions)

        box_wall = walls_next[idxs_arange, boxes_next_tmp[:, 0], boxes_next_tmp[:, 1]]
        box_box = boxes[idxs_arange, boxes_next_tmp[:, 0], boxes_next_tmp[:, 1]]

        # agent -> box -> obstacle
        agent_box_obstacle = agent_box & (box_wall | box_box)
        agent_next[agent_box_obstacle] = agent[agent_box_obstacle]

        # agent -> box -> empty
        agent_box_empty = agent_box & np.logical_not(box_wall | box_box)
        agent_next[agent_box_empty] = agent_next_tmp[agent_box_empty]
        abe_idxs = np.where(agent_box_empty)[0]

        agent_next_idxs_abe = agent_next[agent_box_empty]
        boxes_next_idxs_abe = boxes_next_tmp[agent_box_empty]

        boxes_next[abe_idxs, agent_next_idxs_abe[:, 0], agent_next_idxs_abe[:, 1]] = False
        boxes_next[abe_idxs, boxes_next_idxs_abe[:, 0], boxes_next_idxs_abe[:, 1]] = True

        # agent -> empty
        agent_empty = np.logical_not(agent_wall | agent_box)
        agent_next[agent_empty] = agent_next_tmp[agent_empty]
        boxes_next[agent_empty] = boxes[agent_empty]

        states_next: List[SokobanState] = []
        for idx in range(len(states)):
            state_next: SokobanState = SokobanState(agent_next[idx], boxes_next[idx],
                                                    walls_next[idx])
            states_next.append(state_next)

        transition_costs: List[int] = [1 for _ in range(len(states))]

        return states_next, transition_costs

    def get_dqn(self) -> nn.Module:
        """
        Returns the DQN model for the Sokoban environment.

        Returns:
            nn.Module: DQN model.
        """
        nnet = SokobanDQN(self.enc_dim, 5000, 1000, 4, self.num_moves, True)

        return nnet

    def state_to_real(self, states: List[SokobanState]) -> np.ndarray:
        """
        Converts states to real-world observations.

        Args:
            states (List[SokobanState]): List of states.

        Returns:
            np.ndarray: Real-world observations.
        """
        states_real: np.ndarray = np.zeros((len(states), self.img_dim, self.img_dim, 3))
        for state_idx, state in enumerate(states):
            states_real[state_idx] = self.state_to_rgb(state)

        states_real = states_real.transpose([0, 3, 1, 2])

        return states_real

    def get_env_nnet(self) -> nn.Module:
        """
        Returns the neural network model for the environment.

        Returns:
            nn.Module: Neural network model.
        """
        return EnvModel(self.num_actions_max, self.chan_enc)

    def get_env_nnet_cont(self) -> nn.Module:
        """
        Returns the neural network model for the environment for the continuous setting.

        Returns:
            nn.Module: Neural network model.
        """
        return EnvModelContinuous(self.num_actions_max, 3, self.chan_enc)

    def get_encoder(self) -> nn.Module:
        """
        Returns the encoder model.

        Returns:
            nn.Module: Encoder model.
        """
        return Encoder(3, self.chan_enc)

    def get_decoder(self) -> nn.Module:
        """
        Returns the decoder model.

        Returns:
            nn.Module: Decoder model.
        """
        return Decoder(3, self.chan_enc)

    def is_solved(self, states: List[SokobanState], states_goal: List[SokobanState]) -> np.array:
        """
        Checks if the states are solved.

        Args:
            states (List[SokobanState]): List of states.
            states_goal (List[SokobanState]): List of goal states.

        Returns:
            np.array: Boolean array indicating whether each state is solved.
        """
        boxes = np.stack([state.boxes for state in states], axis=0)
        walls = np.stack([state.walls for state in states], axis=0)

        # agent_goal = np.stack([state.agent for state in states_goal], axis=0)
        boxes_goal = np.stack([state.boxes for state in states_goal], axis=0)
        walls_goal = np.stack([state.walls for state in states_goal], axis=0)

        # agent_same = np.all(agent == agent_goal, axis=1)
        boxes_same = np.all(boxes == boxes_goal, axis=(1, 2))
        walls_same = np.all(walls == walls_goal, axis=(1, 2))

        is_solved_arr = boxes_same & walls_same

        return is_solved_arr

    def generate_start_states(self, num_states: int) -> List[SokobanState]:
        """Generates a list of start states for the Sokoban environment.

        Args:
            num_states (int): Number of start states to generate.

        Returns:
            List[SokobanState]: List of generated start states.
        """
        state_idxs = np.random.randint(0, len(self.states_train), size=num_states)
        states: List[SokobanState] = [self.states_train[idx] for idx in state_idxs]

        step_range: Tuple[int, int] = (0, 100)

        # Initialize
        scrambs: List[int] = list(range(step_range[0], step_range[1] + 1))

        # Scrambles
        step_nums: np.ndarray = np.random.choice(scrambs, num_states)
        step_nums_curr: np.ndarray = np.zeros(num_states)

        # Go backward from goal state
        steps_lt = step_nums_curr < step_nums
        while np.any(steps_lt):
            idxs: np.ndarray = np.where(steps_lt)[0]

            states_to_move: List[SokobanState] = [states[idx] for idx in idxs]
            actions = list(np.random.randint(0, self.num_moves, size=len(states_to_move)))

            states_moved, _ = self.next_state(states_to_move, actions)

            for idx_moved, idx in enumerate(idxs):
                states[idx] = states_moved[idx_moved]

            step_nums_curr[idxs] = step_nums_curr[idxs] + 1
            steps_lt[idxs] = step_nums_curr[idxs] < step_nums[idxs]

        return states

    def get_render_array(self, state: SokobanState) -> np.ndarray:
        """Generates a 2D array representation of the state for rendering.

        Args:
            state (SokobanState): The current state of the environment.

        Returns:
            np.ndarray: 2D array representation of the state.
        """
        state_rendered = np.ones((self.dim, self.dim), dtype=int)
        state_rendered -= state.walls
        state_rendered[state.agent[0], state.agent[1]] = 2
        state_rendered += state.boxes * 2

        return state_rendered

    def state_to_rgb(self, state: SokobanState) -> np.ndarray:
        """Converts the state to an RGB image.

        Args:
            state (SokobanState): The current state of the environment.

        Returns:
            np.ndarray: RGB image representation of the state.
        """
        room = self.get_render_array(state)

        # Assemble the new rgb_room, with all loaded images
        room_rgb = np.zeros(shape=(room.shape[0] * 16, room.shape[1] * 16, 3), dtype=np.uint8)
        for i in range(room.shape[0]):
            x_i = i * 16

            for j in range(room.shape[1]):
                y_j = j * 16
                surfaces_id = room[i, j]

                room_rgb[x_i:(x_i + 16), y_j:(y_j + 16), :] = self._surfaces[surfaces_id]

        room_rgb = room_rgb / 255

        room_rgb = cv2.resize(room_rgb, (self.img_dim, self.img_dim))

        return room_rgb

    def _get_next_idx(self, curr_idxs: np.ndarray, actions: List[int]) -> np.ndarray:
        """Computes the next indices for the agent based on the current indices and actions.

        Args:
            curr_idxs (np.ndarray): Current indices of the agent.
            actions (List[int]): List of actions to be taken.

        Returns:
            np.ndarray: Next indices of the agent.
        """
        actions_np: np.ndarray = np.array(actions)
        next_idxs: np.ndarray = curr_idxs.copy()

        action_idxs = np.where(actions_np == 0)[0]
        next_idxs[action_idxs, 0] = next_idxs[action_idxs, 0] - 1

        action_idxs = np.where(actions_np == 1)[0]
        next_idxs[action_idxs, 0] = next_idxs[action_idxs, 0] + 1

        action_idxs = np.where(actions_np == 2)[0]
        next_idxs[action_idxs, 1] = next_idxs[action_idxs, 1] - 1

        action_idxs = np.where(actions_np == 3)[0]
        next_idxs[action_idxs, 1] = next_idxs[action_idxs, 1] + 1

        next_idxs = np.maximum(next_idxs, 0)
        next_idxs = np.minimum(next_idxs, self.dim - 1)

        return next_idxs


class InteractiveEnv(plt.Axes):

    def __init__(self, env: Sokoban, fig: plt.Figure):
        """Initializes the interactive environment for visualization.

        Args:
            env (Sokoban): The Sokoban environment.
            fig (plt.Figure): The matplotlib figure for rendering.
        """
        self.env = env
        self.state = None

        super(InteractiveEnv, self).__init__(plt.gcf(), [0, 0, 1, 1])

        callbacks = fig.canvas.callbacks.callbacks
        del callbacks["key_press_event"]

        self.figure.canvas.mpl_connect("key_press_event", self._key_press)

        self._get_instance()
        self._update_plot()

        self.move = []

    def _get_instance(self) -> None:
        """Generates a new instance of the environment."""
        states, states_goal, _, _ = self.env.generate_episodes([1000])
        self.state = states[0]
        self.state_goal = states_goal[0]

    def _update_plot(self) -> None:
        """Updates the plot with the current state and goal state."""
        self.clear()
        rendered_im = self.env.state_to_rgb(self.state)
        rendered_im_goal = self.env.state_to_rgb(self.state_goal)

        self.imshow(np.concatenate((rendered_im, rendered_im_goal), axis=1))
        self.figure.canvas.draw()

    def _key_press(self, event: Any) -> None:
        """Handles key press events for interactive control.

        Args:
            event (Any): The key press event.
        """
        if event.key.upper() in "ASDW":
            action: int = -1
            if event.key.upper() == "W":
                action = 0
            if event.key.upper() == "S":
                action = 1
            if event.key.upper() == "A":
                action = 2
            if event.key.upper() == "D":
                action = 3

            self.state = self.env.next_state([self.state], [action])[0][0]
            self._update_plot()
            if self.env.is_solved([self.state], [self.state_goal])[0]:
                print("SOLVED!")
        elif event.key.upper() == "R":
            self._get_instance()
            self._update_plot()
        elif event.key.upper() == "P":
            for _ in range(1000):
                action = self.env.rand_action([self.state])[0]
                self.state = self.env.next_state([self.state], [action])[0][0]
            self._update_plot()


def main() -> None:
    """Main function to run the interactive Sokoban environment."""
    env: Sokoban = Sokoban(10, 4)

    fig = plt.figure(figsize=(5, 5))
    interactive_env = InteractiveEnv(env, fig)
    fig.add_axes(interactive_env)

    plt.show()


if __name__ == "__main__":
    main()
