from copy import copy
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from deepcubeai.environments.environment_abstract import Environment, State
from deepcubeai.environments.puzzlegen.dice_puzzle import DicePuzzle
from deepcubeai.utils.pytorch_models import (
    Conv2dModel,
    FullyConnectedModel,
    ResnetConv2dModel,
    STEThresh,
)


class DigitJumpDQN(nn.Module):
    """Deep Q-Network model for the DigitJump environment.

    Attributes:
        chan_enc (int): Number of channels for the encoder.
        enc_hw (Tuple[int, int]): Height and width of the encoder.
        resnet_chan (int): Number of channels for the ResNet.
        num_resnet_blocks (int): Number of residual blocks.
        num_actions (int): Number of possible actions.
        dqn (nn.Sequential): Sequential model containing the DQN layers.
    """

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, chan_enc: int, enc_hw: Tuple[int, int], resnet_chan: int,
                 num_resnet_blocks: int, num_actions: int, batch_norm: bool):

        super().__init__()

        self.chan_enc: int = chan_enc
        self.enc_hw: Tuple[int, int] = enc_hw
        self.resnet_chan: int = resnet_chan
        self.num_resnet_blocks: int = num_resnet_blocks
        fc_in: int = resnet_chan * enc_hw[0] * enc_hw[1]
        h_dim: int = 3 * fc_in
        use_bias_with_norm = False

        self.dqn = nn.Sequential(
            Conv2dModel(
                chan_enc * 2,
                [resnet_chan],
                [3],
                [1],
                [False],
                ["RELU"],
                group_norms=[1],
                use_bias_with_norm=use_bias_with_norm,
            ),
            ResnetConv2dModel(
                resnet_chan,
                resnet_chan,
                resnet_chan,
                3,
                1,
                num_resnet_blocks,
                batch_norm,
                "RELU",
                group_norm=0,
                use_bias_with_norm=use_bias_with_norm,
            ),
            nn.Flatten(),
            FullyConnectedModel(
                fc_in,
                [h_dim, num_actions],
                [batch_norm] * 2,
                ["RELU"] * 2,
                use_bias_with_norm=use_bias_with_norm,
            ),
        )

    def forward(self, states: Tensor, states_goal: Tensor) -> Tensor:
        """
        Forward pass for the DQN model.

        Args:
            states (Tensor): Current states.
            states_goal (Tensor): Goal states.

        Returns:
            Tensor: Q-values for the given states and goals.
        """
        states_conv = states.view(-1, self.chan_enc, self.enc_hw[0], self.enc_hw[1])
        states_goal_conv = states_goal.view(-1, self.chan_enc, self.enc_hw[0], self.enc_hw[1])

        dqn_input = torch.cat((states_conv.float(), states_goal_conv.float()), dim=1)

        q_values = self.dqn(dqn_input)

        return q_values


class Encoder(nn.Module):
    """Encoder model for the DigitJump environment.

    Attributes:
        chan_in (int): Number of input channels.
        chan_enc (int): Number of channels for the encoder.
        encoder (nn.Sequential): Sequential model containing the encoder layers.
        ste_thresh (STEThresh): Straight-through estimator for thresholding.
    """

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, chan_in: int, chan_enc: int):
        super().__init__()

        self.chan_in: int = chan_in
        self.chan_enc: int = chan_enc
        use_bias_with_norm: bool = False

        self.encoder = nn.Sequential(
            Conv2dModel(
                chan_in,
                [32, chan_enc],
                [4, 2],
                [0, 0],
                [True, False],
                ["RELU", "SIGMOID"],
                strides=[4, 2],
                group_norms=[0, 0],
                use_bias_with_norm=use_bias_with_norm,
            ),
            nn.Flatten(),
        )

        self.ste_thresh = STEThresh()

    def forward(self, states: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the Encoder model.

        Args:
            states (Tensor): Input states.

        Returns:
            Tuple[Tensor, Tensor]: Encoded states and thresholded encoded states.
        """
        encs = self.encoder(states)
        encs_d = self.ste_thresh.apply(encs, 0.5)

        return encs, encs_d


class Decoder(nn.Module):
    """Decoder model for the DigitJump environment.

    Attributes:
        chan_in (int): Number of input channels.
        chan_enc (int): Number of channels for the encoder.
        enc_hw (Tuple[int, int]): Height and width of the encoder.
        decoder_conv (nn.Sequential): Sequential model containing the decoder layers.
    """

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, chan_in: int, chan_enc: int, enc_hw: Tuple[int, int]):
        super().__init__()

        self.chan_in: int = chan_in
        self.chan_enc: int = chan_enc
        self.enc_hw: Tuple[int, int] = enc_hw
        use_bias_with_norm: bool = False

        self.decoder_conv = nn.Sequential(
            Conv2dModel(
                chan_enc,
                [32, 32],
                [2, 4],
                [0, 0],
                [True, False],
                ["RELU", "SIGMOID"],
                strides=[2, 4],
                group_norms=[0, 0],
                transpose=True,
                use_bias_with_norm=use_bias_with_norm,
            ),
            Conv2dModel(32, [chan_in], [1], [0], [False], ["LINEAR"],
                        use_bias_with_norm=use_bias_with_norm),
        )

    def forward(self, encs: Tensor) -> Tensor:
        """
        Forward pass for the Decoder model.

        Args:
            encs (Tensor): Encoded states.

        Returns:
            Tensor: Decoded states.
        """
        decs = torch.reshape(encs, (encs.shape[0], self.chan_enc, self.enc_hw[0], self.enc_hw[1]))
        decs = self.decoder_conv(decs)

        return decs


class EnvModel(nn.Module):
    """Environment model for the DigitJump environment.

    Attributes:
        chan_enc (int): Number of channels for the encoder.
        enc_hw (Tuple[int, int]): Height and width of the encoder.
        resnet_chan (int): Number of channels for the ResNet.
        num_resnet_blocks (int): Number of residual blocks.
        num_actions (int): Number of possible actions.
        mask_net (nn.Sequential): Sequential model containing the mask network layers.
    """

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, chan_enc: int, enc_hw: Tuple[int, int], resnet_chan: int,
                 num_resnet_blocks: int, num_actions: int):
        super().__init__()

        self.chan_enc: int = chan_enc
        self.enc_hw: Tuple[int, int] = enc_hw
        self.resnet_chan: int = resnet_chan
        self.num_resnet_blocks: int = num_resnet_blocks
        self.num_actions: int = num_actions
        use_bias_with_norm: bool = False

        self.mask_net = nn.Sequential(
            Conv2dModel(
                chan_enc + num_actions,
                [resnet_chan],
                [1],
                [0],
                [False],
                ["RELU"],
                group_norms=[1],
                use_bias_with_norm=use_bias_with_norm,
            ),
            ResnetConv2dModel(
                resnet_chan,
                resnet_chan,
                resnet_chan,
                3,
                1,
                num_resnet_blocks,
                True,
                "RELU",
                group_norm=0,
                use_bias_with_norm=use_bias_with_norm,
            ),
            Conv2dModel(
                resnet_chan,
                [chan_enc, chan_enc],
                [1, 1],
                [0, 0],
                [True, False],
                ["RELU", "SIGMOID"],
                group_norms=[0, 0],
                use_bias_with_norm=use_bias_with_norm,
            ),
            nn.Flatten(),
        )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """
        Forward pass for the EnvModel.

        Args:
            states (Tensor): Current states.
            actions (Tensor): Actions to be taken.

        Returns:
            Tensor: Next states.
        """
        states_conv = states.view(-1, self.chan_enc, self.enc_hw[0], self.enc_hw[1])

        actions_oh = F.one_hot(actions.long(), self.num_actions)
        actions_oh = actions_oh.float()

        actions_oh = actions_oh.view(-1, self.num_actions, 1, 1)
        actions_oh = actions_oh.repeat(1, 1, states_conv.shape[2], states_conv.shape[3])

        states_actions = torch.cat((states_conv.float(), actions_oh), dim=1)

        mask = self.mask_net(states_actions)
        states_next = mask

        return states_next


class EnvModelContinuous(nn.Module):
    """Continuous environment model for the DigitJump environment.

    Attributes:
        chan_in (int): Number of input channels.
        chan_enc (int): Number of channels for the encoder.
        resnet_chan (int): Number of channels for the ResNet.
        num_resnet_blocks (int): Number of ResNet blocks.
        num_actions (int): Number of possible actions.
        encoder (nn.Sequential): Sequential model containing the encoder layers.
        env_model (nn.Sequential): Sequential model containing the environment model layers.
    """

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, chan_in: int, chan_enc: int, resnet_chan: int, num_resnet_blocks: int,
                 num_actions: int):
        super().__init__()

        self.chan_in: int = chan_in
        self.chan_enc: int = chan_enc
        self.resnet_chan: int = resnet_chan
        self.num_resnet_blocks: int = num_resnet_blocks
        self.num_actions: int = num_actions
        use_bias_with_norm: bool = False

        self.encoder = nn.Sequential(
            Conv2dModel(
                chan_in,
                [32, chan_enc],
                [4, 2],
                [0, 0],
                [True, False],
                ["RELU", "SIGMOID"],
                strides=[4, 2],
                group_norms=[0, 0],
                use_bias_with_norm=use_bias_with_norm,
            ), )

        self.env_model = nn.Sequential(
            Conv2dModel(
                chan_enc + num_actions,
                [resnet_chan],
                [1],
                [0],
                [False],
                ["RELU"],
                group_norms=[1],
                use_bias_with_norm=use_bias_with_norm,
            ),
            ResnetConv2dModel(
                resnet_chan,
                resnet_chan,
                resnet_chan,
                3,
                1,
                num_resnet_blocks,
                True,
                "RELU",
                group_norm=0,
                use_bias_with_norm=use_bias_with_norm,
            ),
            Conv2dModel(
                resnet_chan,
                [chan_enc, chan_enc],
                [1, 1],
                [0, 0],
                [True, False],
                ["RELU", "SIGMOID"],
                group_norms=[0, 0],
                use_bias_with_norm=use_bias_with_norm,
            ),
            Conv2dModel(
                chan_enc,
                [32, 32],
                [2, 4],
                [0, 0],
                [True, False],
                ["RELU", "SIGMOID"],
                strides=[2, 4],
                group_norms=[0, 0],
                transpose=True,
                use_bias_with_norm=use_bias_with_norm,
            ),
            Conv2dModel(32, [chan_in], [1], [0], [False], ["LINEAR"],
                        use_bias_with_norm=use_bias_with_norm),
        )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """
        Forward pass for the EnvModelContinuous.

        Args:
            states (Tensor): Current states.
            actions (Tensor): Actions to be taken.

        Returns:
            Tensor: Next states.
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


class DigitJumpState(State):
    """State representation for the DigitJump environment.

    Attributes:
        dice_puzzle (DicePuzzle): Instance of the DicePuzzle.
        render_style (str): Rendering style.
        min_sol_len (int): Minimum solution length.
        seed (Optional[int]): Seed for random number generation.
        player_x (int): Player's x-coordinate.
        player_y (int): Player's y-coordinate.
        hash (Optional[int]): Cached hash value.
    """

    __slots__ = [
        "dice_puzzle",
        "render_style",
        "min_sol_len",
        "seed",
        "player_x",
        "player_y",
        "hash",
    ]

    def __init__(self, **kwargs):
        super().__init__()

        self.dice_puzzle: DicePuzzle = None
        self.render_style = "mnist"
        self.min_sol_len = 8
        self.seed = None

        if "dice_puzzle" in kwargs:
            self.dice_puzzle = kwargs["dice_puzzle"]
            self.render_style = self.dice_puzzle.render_style
            self.min_sol_len = self.dice_puzzle.min_sol_len
            self.seed = self.dice_puzzle.seed

        else:

            if "render_style" in kwargs:
                self.render_style = kwargs["render_style"]

            if "min_sol_len" in kwargs:
                self.min_sol_len = kwargs["min_sol_len"]

            if "seed" in kwargs:
                self.seed = kwargs["seed"]

            self.dice_puzzle = DicePuzzle(render_style=self.render_style,
                                          min_sol_len=self.min_sol_len,
                                          seed=self.seed)
            self.dice_puzzle.reset()

        self.player_x = self.dice_puzzle.pos[0]
        self.player_y = self.dice_puzzle.pos[1]
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash((
                self.dice_puzzle.grid,
                self.dice_puzzle.start,
                self.dice_puzzle.end,
                self.dice_puzzle.player_x,
                self.dice_puzzle.player_y,
            ))
        return self.hash

    def __eq__(self, other):
        return (self.dice_puzzle.grid == other.dice_puzzle.grid
                and self.dice_puzzle.start == other.dice_puzzle.start
                and self.dice_puzzle.end == other.dice_puzzle.end
                and self.player_x == other.dice_puzzle.player_x
                and self.player_y == other.dice_puzzle.player_y)

    def move(self, action: int) -> Tuple[float, bool, float, "DigitJumpState"]:
        """
        Moves the player in the given direction.

        Args:
            action (int): Action to be taken.

        Returns:
            Tuple[float, bool, float, DigitJumpState]: Reward, done flag, move cost, and next
              state.
        """
        next_dice_puzzle = copy(self.dice_puzzle)
        reward, done, _ = next_dice_puzzle.step(int(action))
        next_state = DigitJumpState(dice_puzzle=next_dice_puzzle)
        # We assume all all actions have a uniform cost of 1.0
        move_cost = 1.0
        return reward, done, move_cost, next_state

    def get_solution(self) -> List[int]:
        """
        Gets the solution path.

        Returns:
            List[int]: Solution path.
        """
        return self.dice_puzzle.solution

    def get_opt_path_len(self) -> int:
        """
        Gets the optimal path length.

        Returns:
            int: Optimal path length.
        """
        return len(self.dice_puzzle.solution)

    def render_image(self) -> np.ndarray:
        """
        Renders the state as an image.

        Returns:
            np.ndarray: Rendered image.
        """
        return self.dice_puzzle.render()

    def get_grid(self):
        """Gets the grid of the dice puzzle.

        Returns:
            np.ndarray: The grid of the dice puzzle.
        """
        return self.dice_puzzle.grid


class DigitJumpEnvironment(Environment):
    """Environment for the DigitJump game.

    Attributes:
        moves (List[str]): List of possible moves.
        shape (int): Shape of the environment.
        num_moves (int): Number of possible moves.
        chan_enc (int): Number of channels for the encoder.
        enc_hw (Tuple[int, int]): Height and width of the encoder.
    """

    moves: List[str] = ["UP", "RIGHT", "LEFT", "DOWN", "NO-OP"]

    # Actions are: 0 = Up, 1 = Right, 2 = Down, 3 = Left, 4 = No-op

    def __init__(self, shape: int = 64):
        super().__init__()
        self.shape = shape
        self.num_moves = len(self.moves)
        self.chan_enc: int = 12
        enc_h: int = 8
        enc_w: int = 8
        self.enc_hw: Tuple[int, int] = (enc_h, enc_w)

    def get_env_name(self) -> str:
        """Gets the name of the environment.

        Returns:
            str: The name of the environment, "digitjump".
        """
        return "digitjump"

    @property
    def num_actions_max(self):
        """Gets the maximum number of actions.

        Returns:
            int: Maximum number of actions.
        """
        return self.num_moves

    def next_state(self, states: List[DigitJumpState],
                   actions: List[int]) -> Tuple[List[DigitJumpState], List[float]]:
        """Gets the next state and transition cost given the current state and action.

        Args:
            states (List[DigitJumpState]): List of current states.
            actions (List[int]): List of actions to take.

        Returns:
            Tuple[List[DigitJumpState], List[float]]: Next states and transition costs.
        """
        next_states: List[DigitJumpState] = []
        transition_costs: List[float] = []
        for state, action in zip(states, actions):
            _, _, move_cost, next_state = state.move(action)
            next_states.append(next_state)
            transition_costs.append(move_cost)

        return next_states, transition_costs

    def rand_action(self, states: List[DigitJumpState]) -> List[int]:
        """Gets random actions that could be taken in each state.

        Args:
            states (List[DigitJumpState]): List of current states.

        Returns:
            List[int]: List of random actions.
        """
        return list(np.random.randint(0, self.num_actions_max, size=len(states)))

    def is_solved(self, states: List[DigitJumpState],
                  states_goal: List[DigitJumpState]) -> np.ndarray:
        """Checks if the states are solved.

        Args:
            states (List[DigitJumpState]): List of current states.
            states_goal (List[DigitJumpState]): List of goal states.

        Returns:
            np.ndarray: Boolean array indicating whether each state is solved.
        """
        states_np = np.stack([(state.player_x, state.player_y) for state in states], axis=0)
        goal_states_np = np.stack([(goal_state.player_x, goal_state.player_y)
                                   for goal_state in states_goal],
                                  axis=0)

        is_equal = np.equal(states_np, goal_states_np)

        return np.all(is_equal, axis=1)

    def state_to_real(self, states: List[DigitJumpState]) -> np.ndarray:
        """Converts states to real-world observations.

        Args:
            states (List[DigitJumpState]): List of current states.

        Returns:
            np.ndarray: Real-world observations.
        """
        states_real: np.ndarray = np.zeros((len(states), self.shape, self.shape, 3), dtype=float)
        for state_idx, state in enumerate(states):
            states_real[state_idx] = state.render_image() / 255.0

        states_real = states_real.transpose([0, 3, 1, 2])

        return states_real

    def get_dqn(self) -> nn.Module:
        """Gets the DQN model for the environment.

        Returns:
            nn.Module: DQN model.
        """
        resnet_chan: int = 7 * self.chan_enc * 2
        num_resnet_blocks: int = 4
        return DigitJumpDQN(self.chan_enc, self.enc_hw, resnet_chan, num_resnet_blocks,
                            self.num_actions_max, True)

    def get_env_nnet(self) -> nn.Module:
        """Gets the environment neural network model.

        Returns:
            nn.Module: Environment neural network model.
        """
        resnet_chan: int = (self.chan_enc + self.num_actions_max) * 7
        num_resnet_blocks: int = 4
        return EnvModel(self.chan_enc, self.enc_hw, resnet_chan, num_resnet_blocks,
                        self.num_actions_max)

    def get_env_nnet_cont(self) -> nn.Module:
        """Gets the continuous environment neural network model.

        Returns:
            nn.Module: Continuous environment neural network model.
        """
        resnet_chan: int = (self.chan_enc + self.num_actions_max) * 7
        num_resnet_blocks: int = 4
        return EnvModelContinuous(3, self.chan_enc, resnet_chan, num_resnet_blocks,
                                  self.num_actions_max)

    def get_encoder(self) -> nn.Module:
        """Gets the encoder model.

        Returns:
            nn.Module: Encoder model.
        """
        return Encoder(3, self.chan_enc)

    def get_decoder(self) -> nn.Module:
        """Gets the decoder model.

        Returns:
            nn.Module: Decoder model.
        """
        return Decoder(3, self.chan_enc, self.enc_hw)

    def generate_start_states(self,
                              num_states: int,
                              level_seeds: Optional[List[int]] = None) -> List[DigitJumpState]:
        """Generates start states for the environment.

        Args:
            num_states (int): Number of start states to generate.
            level_seeds (Optional[List[int]]): List of seeds for level generation. Defaults to
                None.

        Returns:
            List[DigitJumpState]: List of generated start states.
        """
        start_states = []
        for idx in range(num_states):
            state = DigitJumpState(seed=level_seeds[idx])
            start_states.append(state)

        return start_states

    def get_goals(self, states: List[DigitJumpState]) -> List[DigitJumpState]:
        """Gets the goal states for the input list of states.

        Args:
            states (List[DigitJumpState]): List of current states.

        Returns:
            List[DigitJumpState]: List of goal states.
        """
        goal_states: List[DigitJumpState] = []
        for state in states:
            dice_puzzle_goal_state = DicePuzzle(seed=state.seed)
            dice_puzzle_goal_state.reset()
            dice_puzzle_goal_state.pos = dice_puzzle_goal_state.end
            dice_puzzle_goal_state.already_solved = True
            goal_state = DigitJumpState(dice_puzzle=dice_puzzle_goal_state)
            goal_states.append(goal_state)

        return goal_states
