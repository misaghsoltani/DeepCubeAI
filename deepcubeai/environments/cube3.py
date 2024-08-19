from random import randrange
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch import Tensor, nn

from deepcubeai.environments.environment_abstract import Environment, State
from deepcubeai.utils.pytorch_models import FullyConnectedModel, ResnetModel, STEThresh
from deepcubeai.visualizers.cube3_viz_simple import InteractiveCube


class Cube3FCResnet(nn.Module):
    """Fully Connected ResNet model for Cube3 environment."""

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, input_dim: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool, act_fn: str):
        """
        Initializes the Cube3FCResnet model.

        Args:
            input_dim (int): Input dimension.
            h1_dim (int): Dimension of the first hidden layer.
            resnet_dim (int): Dimension of the ResNet layer.
            num_resnet_blocks (int): Number of residual blocks blocks.
            out_dim (int): Output dimension.
            batch_norm (bool): Whether to use batch normalization.
            act_fn (str): Activation function.
        """
        super().__init__()
        self.first_fc = FullyConnectedModel(input_dim, [h1_dim, resnet_dim], [batch_norm] * 2,
                                            [act_fn] * 2)
        self.resnet = ResnetModel(resnet_dim, num_resnet_blocks, out_dim, batch_norm, act_fn)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Cube3FCResnet model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.first_fc(x)
        x = self.resnet(x)

        return x


class Cube3DQN(nn.Module):
    """Deep Q-Network model for Cube3 environment."""

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, state_dim: int, h1_dim: int, resnet_dim: int, num_res_blocks: int,
                 out_dim: int, batch_norm: bool):
        """
        Initializes the Cube3DQN model.

        Args:
            state_dim (int): Dimension of the state.
            h1_dim (int): Dimension of the first hidden layer.
            resnet_dim (int): Dimension of the ResNet layer.
            num_res_blocks (int): Number of residual blocks.
            out_dim (int): Output dimension.
            batch_norm (bool): Whether to use batch normalization.
        """
        super().__init__()
        self.dqn = Cube3FCResnet(state_dim * 2, h1_dim, resnet_dim, num_res_blocks, out_dim,
                                 batch_norm, "RELU")

    def forward(self, states: Tensor, states_goal: Tensor) -> Tensor:
        """
        Forward pass of the Cube3DQN model.

        Args:
            states (Tensor): Current states.
            states_goal (Tensor): Goal states.

        Returns:
            Tensor: Q-values for the given states and goals.
        """
        x = self.dqn(torch.cat((states.float(), states_goal.float()), dim=1))

        return x


class Encoder(nn.Module):
    """Encoder model for Cube3 environment."""

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, chan_in: int):
        """
        Initializes the Encoder model.

        Args:
            chan_in (int): Number of input channels.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(32 * 32 * chan_in),
            FullyConnectedModel(32 * 32 * chan_in, [400], [False], ["SIGMOID"]),
        )

        self.ste_thresh = STEThresh()

    def forward(self, states: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the Encoder model.

        Args:
            states (Tensor): Input states.

        Returns:
            Tuple[Tensor, Tensor]: Encoded states and rounded encoded states.
        """
        encs = self.encoder(states)
        encs_d = self.ste_thresh.apply(encs, 0.5)

        return encs, encs_d


class Decoder(nn.Module):
    """Decoder model for Cube3 environment."""

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, chan_in: int):
        """
        Initializes the Decoder model.

        Args:
            chan_in (int): Number of input channels.
        """
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Flatten(),
            FullyConnectedModel(400, [32 * 32 * chan_in], [False], ["LINEAR"]),
        )

    def forward(self, encs: Tensor) -> Tensor:
        """
        Forward pass of the Decoder model.

        Args:
            encs (Tensor): Encoded states.

        Returns:
            Tensor: Decoded states.
        """
        decs = torch.reshape(self.decoder(encs), (encs.shape[0], 6, 32, 32))

        return decs


class EnvModel(nn.Module):
    """Environment model for Cube3 environment."""

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, state_dim: int, num_actions: int):
        """
        Initializes the EnvModel.

        Args:
            state_dim (int): Dimension of the state.
            num_actions (int): Number of possible actions.
        """
        super().__init__()
        self.num_actions = num_actions

        self.env_model = nn.Sequential(
            FullyConnectedModel(
                state_dim + num_actions,
                [500, 500, 500, state_dim],
                [True, True, True, False],
                ["RELU", "RELU", "RELU", "SIGMOID"],
            ), )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """
        Forward pass of the EnvModel.

        Args:
            states (Tensor): Current states.
            actions (Tensor): Actions to be taken.

        Returns:
            Tensor: Next states.
        """
        actions_oh = F.one_hot(actions.long(), self.num_actions)
        actions_oh = actions_oh.float()
        actions_oh = actions_oh.view(-1, self.num_actions)

        states_actions = torch.cat((states.float(), actions_oh), dim=1)

        states_next = self.env_model(states_actions)

        return states_next


class EnvModelContinuous(nn.Module):
    """Continuous environment model for Cube3 environment."""

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, state_dim: int, chan_in: int, num_actions: int):
        """
        Initializes the EnvModelContinuous.

        Args:
            state_dim (int): Dimension of the state.
            chan_in (int): Number of input channels.
            num_actions (int): Number of possible actions.
        """
        super().__init__()
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(32 * 32 * chan_in),
            FullyConnectedModel(32 * 32 * chan_in, [state_dim], [False], ["SIGMOID"]),
        )

        self.env_model = nn.Sequential(
            FullyConnectedModel(
                state_dim + num_actions,
                [500, 500, 500, state_dim],
                [True, True, True, False],
                ["RELU", "RELU", "RELU", "SIGMOID"],
            ), )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            FullyConnectedModel(state_dim, [32 * 32 * chan_in], [False], ["LINEAR"]),
        )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """
        Forward pass of the EnvModelContinuous.

        Args:
            states (Tensor): Current states.
            actions (Tensor): Actions to be taken.

        Returns:
            Tensor: Next states.
        """
        states_enc = self.encoder(states.float())

        actions_oh = F.one_hot(actions.long(), self.num_actions)
        actions_oh = actions_oh.float()
        actions_oh = actions_oh.view(-1, self.num_actions)

        states_actions = torch.cat((states_enc, actions_oh), dim=1)

        states_next_enc = self.env_model(states_actions)
        states_next = torch.reshape(self.decoder(states_next_enc),
                                    (states_next_enc.shape[0], 6, 32, 32))

        return states_next


class Cube3State(State):
    """State representation for Cube3 environment."""

    __slots__ = ["colors"]

    def __init__(self, colors: np.ndarray):
        """
        Initializes the Cube3State.

        Args:
            colors (np.ndarray): Colors of the cube.
        """
        super().__init__()
        self.colors: np.ndarray = colors


class Cube3(Environment):
    """Cube3 environment class."""

    moves: List[str] = [f"{f}{n}" for f in ["U", "D", "L", "R", "B", "F"] for n in [-1, 1]]
    moves_rev: List[str] = [f"{f}{n}" for f in ["U", "D", "L", "R", "B", "F"] for n in [1, -1]]

    def __init__(self):
        """
        Initializes the Cube3 environment.

        Args:
            do_action_triples (bool): Whether to use action triples.
        """
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 3

        self.do_action_triples: bool = False
        self.num_moves = 12

        # solved state
        self.goal_colors: np.ndarray = np.arange(0, (self.cube_len**2) * 6, 1, dtype=self.dtype)

        # get idxs changed for moves
        self.rotate_idxs_new: Dict[str, np.ndarray]
        self.rotate_idxs_old: Dict[str, np.ndarray]

        self.adj_faces: Dict[int, np.ndarray]
        self._get_adj()

        self.rotate_idxs_new, self.rotate_idxs_old = self._compute_rotation_idxs(
            self.cube_len, self.moves)

    def get_env_name(self) -> str:
        """Gets the name of the environment.

        Returns:
            str: The name of the environment, "cube3".
        """
        return "cube3"

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
        Returns random actions for the given states.

        Args:
            states (List[State]): List of states.

        Returns:
            List[int]: List of random actions.
        """
        return list(np.random.randint(0, self.num_moves, size=len(states)))

    def next_state(self, states: List[Cube3State],
                   actions_l: List[int]) -> Tuple[List[Cube3State], List[float]]:
        """
        Returns the next states and transition costs given the current states and actions.

        Args:
            states (List[Cube3State]): List of current states.
            actions_l (List[int]): List of actions.

        Returns:
            Tuple[List[Cube3State], List[float]]: Next states and transition costs.
        """
        states_np = np.stack([x.colors for x in states], axis=0)

        states_next_np = np.zeros(states_np.shape, dtype=self.dtype)
        tcs_np: np.array = np.zeros(len(states))
        actions: np.array = np.array(actions_l)
        for action in np.unique(actions):
            action_idxs = actions == action
            states_np_act = states_np[actions == action]

            states_next_np_act, tcs_act = self._move_np(states_np_act, action)

            states_next_np[action_idxs] = states_next_np_act
            tcs_np[action_idxs] = np.array(tcs_act)

        states_next: List[Cube3State] = [Cube3State(x) for x in list(states_next_np)]
        transition_costs = list(tcs_np)

        return states_next, transition_costs

    def is_solved(self, states: List[Cube3State], states_goal: List[Cube3State]) -> np.array:
        """
        Checks if the given states are solved.

        Args:
            states (List[Cube3State]): List of current states.
            states_goal (List[Cube3State]): List of goal states.

        Returns:
            np.array: Boolean array indicating whether each state is solved.
        """
        states_np = np.stack([state.colors for state in states], axis=0)
        states_goal_np = np.stack([state.colors for state in states_goal], axis=0)

        is_equal = np.equal(states_np, states_goal_np)

        return np.all(is_equal, axis=1)

    def state_to_real(self, states: List[Cube3State]) -> np.ndarray:
        """
        Converts the given states to real-world observations.

        Args:
            states (List[Cube3State]): List of states.

        Returns:
            np.ndarray: Real-world observations.
        """
        # initialize
        fig = plt.figure(figsize=(0.32, 0.32))
        viz = InteractiveCube(3, self.generate_start_states(1)[0].colors)
        fig.add_axes(viz)
        canvas = FigureCanvas(fig)
        width, height = fig.get_size_inches() * fig.get_dpi()
        width = int(width)
        height = int(height)

        states_img: np.ndarray = np.zeros((len(states), 32, 32, 6))
        for state_idx, state in enumerate(states):
            # create image
            viz.new_state(state.colors)

            viz.set_rot(0)
            canvas.draw()
            image1 = (
                np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(width, height, 3) /
                255)

            viz.set_rot(1)
            canvas.draw()
            image2 = (
                np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(width, height, 3) /
                255)

            states_img[state_idx] = np.concatenate((image1, image2), axis=2)

        plt.close(fig)

        states_img = np.transpose(states_img, (0, 3, 1, 2))

        return states_img

    def get_dqn(self) -> nn.Module:
        """
        Returns the DQN model for the Cube3 environment.

        Returns:
            nn.Module: DQN model.
        """
        nnet = Cube3DQN(400, 5000, 1000, 4, self.num_actions_max, True)

        return nnet

    def get_env_nnet(self) -> nn.Module:
        """
        Returns the environment neural network model.

        Returns:
            nn.Module: Environment neural network model.
        """
        return EnvModel(400, self.num_actions_max)

    def get_env_nnet_cont(self) -> nn.Module:
        """
        Returns the continuous environment neural network model.

        Returns:
            nn.Module: Continuous environment neural network model.
        """
        return EnvModelContinuous(400, 6, self.num_actions_max)

    def get_encoder(self) -> nn.Module:
        """
        Returns the encoder model.

        Returns:
            nn.Module: Encoder model.
        """
        return Encoder(6)

    def get_decoder(self) -> nn.Module:
        """
        Returns the decoder model.

        Returns:
            nn.Module: Decoder model.
        """
        return Decoder(6)

    def generate_start_states(self, num_states: int) -> List[Cube3State]:
        """
        Generates the start states for the Cube3 environment.

        Args:
            num_states (int): Number of start states to generate.

        Returns:
            List[Cube3State]: List of start states.
        """
        assert num_states > 0
        assert (self.fixed_actions
                ), "Environments without fixed actions must implement their own method"

        backwards_range: Tuple[int, int] = (100, 200)

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = len(self.moves)

        # Get goal states
        states_np: np.ndarray = self._generate_canonical_goal_states(num_states, np_format=True)

        # Scrambles
        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        moves_lt = num_back_moves < scramble_nums
        while np.any(moves_lt):
            idxs: np.ndarray = np.where(moves_lt)[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            states_np[idxs], _ = self._move_np(states_np[idxs], move)

            num_back_moves[idxs] = num_back_moves[idxs] + 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

        states: List[Cube3State] = [Cube3State(x) for x in list(states_np)]

        return states

    def _generate_canonical_goal_states(self,
                                        num_states: int,
                                        np_format: bool = False
                                        ) -> Union[List[Cube3State], np.ndarray]:
        """
        Generates the canonical goal states.

        Args:
            num_states (int): Number of goal states to generate.
            np_format (bool): Whether to return the states in numpy format.

        Returns:
            Union[List[Cube3State], np.ndarray]: List of goal states or numpy array of goal states.
        """
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_colors.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[Cube3State] = [
                Cube3State(self.goal_colors.copy()) for _ in range(num_states)
            ]

        return solved_states

    def _move_np(self, states_np: np.ndarray, action: int) -> Tuple[np.ndarray, List[float]]:
        """
        Applies the given action to the states.

        Args:
            states_np (np.ndarray): Current states.
            action (int): Action to be taken.

        Returns:
            Tuple[np.ndarray, List[float]]: Next states and transition costs.
        """
        states_next_np: np.ndarray = states_np.copy()

        if self.do_action_triples:
            actions = list(self.action_triples[action])
        else:
            actions = [action]

        for action_part in actions:
            action_str: str = self.moves[action_part]
            states_next_np[:, self.rotate_idxs_new[
                action_str]] = states_next_np[:, self.rotate_idxs_old[action_str]]

        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs

    def _get_adj(self) -> None:
        """Initializes the adjacency faces for the Cube3 environment.

        This method sets up the adjacency relationships between the faces of the cube.
        The faces are represented by integers:
        - WHITE: 0
        - YELLOW: 1
        - BLUE: 2
        - GREEN: 3
        - ORANGE: 4
        - RED: 5

        The adjacency relationships are stored in the `adj_faces` attribute.
        """
        self.adj_faces: Dict[int, np.ndarray] = {
            0: np.array([2, 5, 3, 4]),
            1: np.array([2, 4, 3, 5]),
            2: np.array([0, 4, 1, 5]),
            3: np.array([0, 5, 1, 4]),
            4: np.array([0, 3, 1, 2]),
            5: np.array([0, 2, 1, 3]),
        }

    def _compute_rotation_idxs(
            self, cube_len: int,
            moves: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Computes the rotation indices for the cube faces based on the given moves.

        Args:
            cube_len (int): The length of one side of the cube.
            moves (List[str]): A list of moves to be applied to the cube. Each move is represented
                as a string, where the first character is the face ('U', 'D', 'L', 'R', 'B', 'F')
                and the second character is the direction (1 or -1).

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: Two dictionaries containing the
                new and old rotation indices for each move.
                - The keys are the move strings.
                - The values are numpy arrays of flattened indices representing the positions of
                    the colors on the cube faces.
        """
        rotate_idxs_new: Dict[str, np.ndarray] = {}
        rotate_idxs_old: Dict[str, np.ndarray] = {}

        for move in moves:
            f: str = move[0]
            sign: int = int(move[1:])

            rotate_idxs_new[move] = np.array([], dtype=int)
            rotate_idxs_old[move] = np.array([], dtype=int)

            colors = np.zeros((6, cube_len, cube_len), dtype=np.int64)
            colors_new = np.copy(colors)

            # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5

            adj_idxs = {
                0: {
                    2: [range(0, cube_len), cube_len - 1],
                    3: [range(0, cube_len), cube_len - 1],
                    4: [range(0, cube_len), cube_len - 1],
                    5: [range(0, cube_len), cube_len - 1],
                },
                1: {
                    2: [range(0, cube_len), 0],
                    3: [range(0, cube_len), 0],
                    4: [range(0, cube_len), 0],
                    5: [range(0, cube_len), 0],
                },
                2: {
                    0: [0, range(0, cube_len)],
                    1: [0, range(0, cube_len)],
                    4: [cube_len - 1, range(cube_len - 1, -1, -1)],
                    5: [0, range(0, cube_len)],
                },
                3: {
                    0: [cube_len - 1, range(0, cube_len)],
                    1: [cube_len - 1, range(0, cube_len)],
                    4: [0, range(cube_len - 1, -1, -1)],
                    5: [cube_len - 1, range(0, cube_len)],
                },
                4: {
                    0: [range(0, cube_len), cube_len - 1],
                    1: [range(cube_len - 1, -1, -1), 0],
                    2: [0, range(0, cube_len)],
                    3: [cube_len - 1, range(cube_len - 1, -1, -1)],
                },
                5: {
                    0: [range(0, cube_len), 0],
                    1: [range(cube_len - 1, -1, -1), cube_len - 1],
                    2: [cube_len - 1, range(0, cube_len)],
                    3: [0, range(cube_len - 1, -1, -1)],
                },
            }
            face_dict = {"U": 0, "D": 1, "L": 2, "R": 3, "B": 4, "F": 5}
            face = face_dict[f]

            faces_to = self.adj_faces[face]
            if sign == 1:
                faces_from = faces_to[(np.arange(0, len(faces_to)) + 1) % len(faces_to)]
            else:
                faces_from = faces_to[(np.arange(
                    len(faces_to) - 1,
                    len(faces_to) - 1 + len(faces_to))) % len(faces_to)]

            cubes_idxs = [
                [0, range(0, cube_len)],
                [range(0, cube_len), cube_len - 1],
                [cube_len - 1, range(cube_len - 1, -1, -1)],
                [range(cube_len - 1, -1, -1), 0],
            ]
            cubes_to = np.array([0, 1, 2, 3])
            if sign == 1:
                cubes_from = cubes_to[(np.arange(
                    len(cubes_to) - 1,
                    len(cubes_to) - 1 + len(cubes_to))) % len(cubes_to)]
            else:
                cubes_from = cubes_to[(np.arange(0, len(cubes_to)) + 1) % len(cubes_to)]

            for i in range(4):
                idxs_new = [[idx1, idx2]
                            for idx1 in np.array([cubes_idxs[cubes_to[i]][0]]).flatten()
                            for idx2 in np.array([cubes_idxs[cubes_to[i]][1]]).flatten()]
                idxs_old = [[idx1, idx2]
                            for idx1 in np.array([cubes_idxs[cubes_from[i]][0]]).flatten()
                            for idx2 in np.array([cubes_idxs[cubes_from[i]][1]]).flatten()]

                for idx_new, idx_old in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face, idx_new[0], idx_new[1]),
                                                        colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face, idx_old[0], idx_old[1]),
                                                        colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

            # Rotate adjacent faces
            face_idxs = adj_idxs[face]
            # pylint: disable=consider-using-enumerate
            for i in range(0, len(faces_to)):
                face_to = faces_to[i]
                face_from = faces_from[i]
                idxs_new = [[idx1, idx2] for idx1 in np.array([face_idxs[face_to][0]]).flatten()
                            for idx2 in np.array([face_idxs[face_to][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([face_idxs[face_from][0]]).flatten()
                            for idx2 in np.array([face_idxs[face_from][1]]).flatten()]
                for idx_new, idx_old in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face_to, idx_new[0], idx_new[1]),
                                                        colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face_from, idx_old[0], idx_old[1]),
                                                        colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

            # pylint: disable=consider-using-enumerate

        return rotate_idxs_new, rotate_idxs_old


class Cube3Triples(Cube3):
    """Cube3Triples environment class."""

    moves: List[str] = [f"{f}{n}" for f in ["U", "D", "L", "R", "B", "F"] for n in [-1, 1]]
    moves_rev: List[str] = [f"{f}{n}" for f in ["U", "D", "L", "R", "B", "F"] for n in [1, -1]]

    def __init__(self):
        """
        Initializes the Cube3Triples environment.
        """
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 3

        self.do_action_triples: bool = True

        self.num_moves = 12**3
        self.action_triples: List[Tuple] = []
        for i in range(12):
            for j in range(12):
                for k in range(12):
                    self.action_triples.append((i, j, k))

        # solved state
        self.goal_colors: np.ndarray = np.arange(0, (self.cube_len**2) * 6, 1, dtype=self.dtype)

        # get idxs changed for moves
        self.rotate_idxs_new: Dict[str, np.ndarray]
        self.rotate_idxs_old: Dict[str, np.ndarray]

        self.adj_faces: Dict[int, np.ndarray]
        self._get_adj()

        self.rotate_idxs_new, self.rotate_idxs_old = self._compute_rotation_idxs(
            self.cube_len, self.moves)

    def get_env_name(self) -> str:
        """Gets the name of the environment.

        Returns:
            str: The name of the environment, "cube3_triples".
        """
        return "cube3_triples"
