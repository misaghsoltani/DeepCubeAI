from __future__ import annotations

from copy import copy
from typing import cast

import numpy as np
from numpy import float32, uint8
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from deepcubeai.environments.environment_abstract import Environment, State
from deepcubeai.environments.puzzlegen.dice_puzzle import DicePuzzle
from deepcubeai.utils.pytorch_models import Conv2dModel, FullyConnectedModel, ResnetConv2dModel, STEThresh


class DigitJumpDQN(nn.Module):
    """Deep Q-Network model for the DigitJump environment.

    Attributes:
        chan_enc (int): Number of channels for the encoder.
        enc_hw (tuple[int, int]): Height and width of the encoder.
        resnet_chan (int): Number of channels for the ResNet.
        num_resnet_blocks (int): Number of residual blocks.
        num_actions (int): Number of possible actions.
        dqn (nn.Sequential): Sequential model containing the DQN layers.
    """

    def __init__(
        self,
        chan_enc: int,
        enc_hw: tuple[int, int],
        resnet_chan: int,
        num_resnet_blocks: int,
        num_actions: int,
        batch_norm: bool,
    ) -> None:
        super().__init__()

        self.chan_enc: int = chan_enc
        self.enc_hw: tuple[int, int] = enc_hw
        self.resnet_chan: int = resnet_chan
        self.num_resnet_blocks: int = num_resnet_blocks
        fc_in: int = resnet_chan * enc_hw[0] * enc_hw[1]
        h_dim: int = 3 * fc_in
        use_bias_with_norm = False

        self.dqn: nn.Sequential = nn.Sequential(
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
                fc_in, [h_dim, num_actions], [batch_norm] * 2, ["RELU"] * 2, use_bias_with_norm=use_bias_with_norm
            ),
        )

    def forward(self, states: Tensor, states_goal: Tensor) -> Tensor:
        """Forward pass for the DQN model.

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

    def __init__(self, chan_in: int, chan_enc: int) -> None:
        super().__init__()

        self.chan_in: int = chan_in
        self.chan_enc: int = chan_enc
        use_bias_with_norm: bool = False

        self.encoder: nn.Sequential = nn.Sequential(
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

    # Use typed helper for straight-through estimator

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for the Encoder model.

        Args:
            states (Tensor): Input states.

        Returns:
            tuple[Tensor, Tensor]: Encoded states and thresholded encoded states.
        """
        encs = self.encoder(states)
        encs_d = cast(Tensor, STEThresh.apply(encs, 0.5))
        return encs, encs_d


class Decoder(nn.Module):
    """Decoder model for the DigitJump environment.

    Attributes:
        chan_in (int): Number of input channels.
        chan_enc (int): Number of channels for the encoder.
        enc_hw (tuple[int, int]): Height and width of the encoder.
        decoder_conv (nn.Sequential): Sequential model containing the decoder layers.
    """

    def __init__(self, chan_in: int, chan_enc: int, enc_hw: tuple[int, int]) -> None:
        super().__init__()

        self.chan_in: int = chan_in
        self.chan_enc: int = chan_enc
        self.enc_hw: tuple[int, int] = enc_hw
        use_bias_with_norm: bool = False

        self.decoder_conv: nn.Sequential = nn.Sequential(
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
            Conv2dModel(32, [chan_in], [1], [0], [False], ["LINEAR"], use_bias_with_norm=use_bias_with_norm),
        )

    def forward(self, encs: Tensor) -> Tensor:
        """Forward pass for the Decoder model.

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
        enc_hw (tuple[int, int]): Height and width of the encoder.
        resnet_chan (int): Number of channels for the ResNet.
        num_resnet_blocks (int): Number of residual blocks.
        num_actions (int): Number of possible actions.
        mask_net (nn.Sequential): Sequential model containing the mask network layers.
    """

    def __init__(
        self, chan_enc: int, enc_hw: tuple[int, int], resnet_chan: int, num_resnet_blocks: int, num_actions: int
    ) -> None:
        super().__init__()

        self.chan_enc: int = chan_enc
        self.enc_hw: tuple[int, int] = enc_hw
        self.resnet_chan: int = resnet_chan
        self.num_resnet_blocks: int = num_resnet_blocks
        self.num_actions: int = num_actions
        use_bias_with_norm: bool = False
        self.mask_net: nn.Sequential = nn.Sequential(
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
        """Forward pass for the EnvModel.

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

    def __init__(self, chan_in: int, chan_enc: int, resnet_chan: int, num_resnet_blocks: int, num_actions: int) -> None:
        super().__init__()

        self.chan_in: int = chan_in
        self.chan_enc: int = chan_enc
        self.resnet_chan: int = resnet_chan
        self.num_resnet_blocks: int = num_resnet_blocks
        self.num_actions: int = num_actions
        use_bias_with_norm: bool = False

        self.encoder: nn.Sequential = nn.Sequential(
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
            )
        )

        self.env_model: nn.Sequential = nn.Sequential(
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
            Conv2dModel(32, [chan_in], [1], [0], [False], ["LINEAR"], use_bias_with_norm=use_bias_with_norm),
        )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """Forward pass for the EnvModelContinuous.

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

    __slots__ = ["dice_puzzle", "render_style", "min_sol_len", "seed", "player_x", "player_y", "hash"]

    def __init__(
        self,
        *,
        dice_puzzle: DicePuzzle | None = None,
        render_style: str = "mnist",
        min_sol_len: int = 8,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        # Initialize from provided arguments
        self.seed: int | None = seed
        self.render_style: str = render_style
        self.min_sol_len: int = min_sol_len
        self.dice_puzzle: DicePuzzle | None = dice_puzzle

        if self.dice_puzzle is None:
            # Create a new puzzle when one isn't supplied
            self.dice_puzzle = DicePuzzle(render_style=self.render_style, min_sol_len=self.min_sol_len, seed=self.seed)
            self.dice_puzzle.reset()
        else:
            # Sync attributes from provided puzzle
            self.render_style = self.dice_puzzle.render_style
            self.min_sol_len = self.dice_puzzle.min_sol_len
            seed_attr = getattr(self.dice_puzzle, "seed", None)
            if seed_attr is not None and not callable(seed_attr):
                self.seed = seed_attr

        if self.dice_puzzle is not None and hasattr(self.dice_puzzle, "pos") and self.dice_puzzle.pos is not None:
            self.player_x: int = self.dice_puzzle.pos[0]
            self.player_y: int = self.dice_puzzle.pos[1]
        else:
            self.player_x = 0
            self.player_y = 0
        self.hash: int | None = None

    def __hash__(self) -> int:
        """Hash method for the DigitJumpState."""
        if self.hash is None:
            if self.dice_puzzle is not None:
                # Convert grid to tuple for hashing if it's an array
                grid_tuple = (
                    tuple(self.dice_puzzle.grid.flat)
                    if hasattr(self.dice_puzzle.grid, "flat") and self.dice_puzzle.grid is not None
                    else self.dice_puzzle.grid
                )
                start_tuple = (
                    tuple(self.dice_puzzle.start)
                    if hasattr(self.dice_puzzle.start, "__iter__") and self.dice_puzzle.start is not None
                    else self.dice_puzzle.start
                )
                end_tuple = (
                    tuple(self.dice_puzzle.end)
                    if hasattr(self.dice_puzzle.end, "__iter__") and self.dice_puzzle.end is not None
                    else self.dice_puzzle.end
                )

                self.hash = hash((grid_tuple, start_tuple, end_tuple, self.player_x, self.player_y))
            else:
                self.hash = hash((self.player_x, self.player_y))
        return self.hash

    def __eq__(self, other: object) -> bool:
        """Equality method for the DigitJumpState."""
        if not isinstance(other, DigitJumpState):
            return False
        if (
            self.dice_puzzle is not None
            and other.dice_puzzle is not None
            and hasattr(self.dice_puzzle, "grid")
            and hasattr(other.dice_puzzle, "grid")
        ):
            try:
                grid1 = getattr(self.dice_puzzle, "grid", None)
                grid2 = getattr(other.dice_puzzle, "grid", None)
                start1 = getattr(self.dice_puzzle, "start", None)
                start2 = getattr(other.dice_puzzle, "start", None)
                end1 = getattr(self.dice_puzzle, "end", None)
                end2 = getattr(other.dice_puzzle, "end", None)

                grids_equal = False
                if grid1 is None and grid2 is None:
                    grids_equal = True
                elif grid1 is not None and grid2 is not None:
                    try:
                        # Prefer numpy array comparison when available
                        grids_equal = np.array_equal(grid1, grid2)
                    except Exception:
                        grids_equal = bool(grid1 == grid2)

                return (
                    grids_equal
                    and start1 == start2
                    and end1 == end2
                    and self.player_x == other.player_x
                    and self.player_y == other.player_y
                )
            except Exception:
                return self.player_x == other.player_x and self.player_y == other.player_y
        elif self.dice_puzzle is None and other.dice_puzzle is None:
            return self.player_x == other.player_x and self.player_y == other.player_y
        else:
            return False

    def move(self, action: int) -> tuple[float, bool, float, DigitJumpState]:
        """Moves the player in the given direction.

        Args:
            action (int): Action to be taken.

        Returns:
            tuple[float, bool, float, DigitJumpState]: Reward, done flag, move cost, and next
              state.
        """
        next_dice_puzzle: DicePuzzle | None = None
        raw_reward: float | int = 0.0
        raw_done: bool = False
        if self.dice_puzzle is not None:
            next_dice_puzzle = copy(self.dice_puzzle)
            try:
                rr, rd, _ = next_dice_puzzle.step(int(action))
                raw_reward, raw_done = rr, rd
            except Exception:
                # keep defaults
                pass

        # Ensure reward is float and done is bool
        reward_float = float(raw_reward)
        done = bool(raw_done)

        next_state = DigitJumpState(dice_puzzle=next_dice_puzzle)
        move_cost = 1.0
        return reward_float, done, move_cost, next_state

    def get_solution(self) -> list[int]:
        """Gets the solution path.

        Returns:
            list[int]: Solution path.
        """
        if (
            self.dice_puzzle is not None
            and hasattr(self.dice_puzzle, "solution")
            and self.dice_puzzle.solution is not None
        ):
            return self.dice_puzzle.solution
        return []

    def get_opt_path_len(self) -> int:
        """Gets the optimal path length.

        Returns:
            int: Optimal path length.
        """
        if (
            self.dice_puzzle is not None
            and hasattr(self.dice_puzzle, "solution")
            and self.dice_puzzle.solution is not None
        ):
            return len(self.dice_puzzle.solution)
        return 0

    def render_image(self) -> NDArray[uint8]:
        """Renders the state as an image.

        Returns:
            NDArray: Rendered image.
        """
        if self.dice_puzzle is not None:
            return self.dice_puzzle.render()
        return np.zeros((64, 64, 3), dtype=uint8)

    def get_grid(self) -> NDArray[uint8]:
        """Gets the grid of the dice puzzle.

        Returns:
            NDArray: The grid of the dice puzzle.
        """
        if self.dice_puzzle is not None and self.dice_puzzle.grid is not None:
            return self.dice_puzzle.grid
        return np.zeros((8, 8), dtype=uint8)


class DigitJumpEnvironment(Environment):
    """Environment for the DigitJump game.

    Attributes:
        moves (list[str]): List of possible moves.
        shape (int): Shape of the environment.
        num_moves (int): Number of possible moves.
        chan_enc (int): Number of channels for the encoder.
        enc_hw (tuple[int, int]): Height and width of the encoder.
    """

    moves: list[str] = ["UP", "RIGHT", "LEFT", "DOWN", "NO-OP"]

    # Actions are: 0 = Up, 1 = Right, 2 = Down, 3 = Left, 4 = No-op

    def __init__(self, shape: int = 64) -> None:
        super().__init__()
        self.shape = shape
        self.num_moves: int = len(self.moves)
        self.chan_enc: int = 12
        enc_h: int = 8
        enc_w: int = 8
        self.enc_hw: tuple[int, int] = (enc_h, enc_w)

    @property
    def env_name(self) -> str:
        """Gets the name of the environment.

        Returns:
            str: The name of the environment, "digitjump".
        """
        return "digitjump"

    @property
    def num_actions_max(self) -> int:
        """Gets the maximum number of actions.

        Returns:
            int: Maximum number of actions.
        """
        return self.num_moves

    @staticmethod
    def next_state(states: list[State], actions: list[int]) -> tuple[list[State], list[float]]:
        """Gets the next state and transition cost given the current state and action.

        Args:
            states (list[State]): List of current states.
            actions (list[int]): List of actions to take.

        Returns:
            tuple[list[State], list[float]]: Next states and transition costs.
        """
        digit_jump_states = [state for state in states if isinstance(state, DigitJumpState)]
        next_states: list[State] = []
        transition_costs: list[float] = []
        for state, action in zip(digit_jump_states, actions, strict=False):
            _, _, move_cost, next_state = state.move(action)
            next_states.append(next_state)
            transition_costs.append(move_cost)

        return next_states, transition_costs

    def rand_action(self, states: list[State]) -> list[int]:
        """Gets random actions that could be taken in each state.

        Args:
            states (list[State]): List of current states.

        Returns:
            list[int]: List of random actions.
        """
        return list(np.random.randint(0, self.num_actions_max, size=len(states)))

    @staticmethod
    def is_solved(states: list[State], states_goal: list[State]) -> NDArray[np.bool_]:
        """Checks if the states are solved.

        Args:
            states (list[State]): List of current states.
            states_goal (list[State]): List of goal states.

        Returns:
            NDArray: Boolean array indicating whether each state is solved.
        """
        digit_jump_states = [state for state in states if isinstance(state, DigitJumpState)]
        digit_jump_goal_states = [state for state in states_goal if isinstance(state, DigitJumpState)]

        states_np = np.stack([(state.player_x, state.player_y) for state in digit_jump_states], axis=0)
        goal_states_np = np.stack(
            [(goal_state.player_x, goal_state.player_y) for goal_state in digit_jump_goal_states], axis=0
        )

        is_equal = np.equal(states_np, goal_states_np)
        return cast(NDArray[np.bool_], np.all(is_equal, axis=1))

    def state_to_real(self, states: list[State]) -> NDArray[float32]:
        """Converts states to real-world observations.

        Args:
            states (list[State]): List of current states.

        Returns:
            NDArray: Real-world observations.
        """
        digit_jump_states = [state for state in states if isinstance(state, DigitJumpState)]
        states_real: NDArray[float32] = np.zeros((len(digit_jump_states), self.shape, self.shape, 3), dtype=float32)
        for state_idx, state in enumerate(digit_jump_states):
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
        return DigitJumpDQN(self.chan_enc, self.enc_hw, resnet_chan, num_resnet_blocks, self.num_actions_max, True)

    def get_env_nnet(self) -> nn.Module:
        """Gets the environment neural network model.

        Returns:
            nn.Module: Environment neural network model.
        """
        resnet_chan: int = (self.chan_enc + self.num_actions_max) * 7
        num_resnet_blocks: int = 4
        return EnvModel(self.chan_enc, self.enc_hw, resnet_chan, num_resnet_blocks, self.num_actions_max)

    def get_env_nnet_cont(self) -> nn.Module:
        """Gets the continuous environment neural network model.

        Returns:
            nn.Module: Continuous environment neural network model.
        """
        resnet_chan: int = (self.chan_enc + self.num_actions_max) * 7
        num_resnet_blocks: int = 4
        return EnvModelContinuous(3, self.chan_enc, resnet_chan, num_resnet_blocks, self.num_actions_max)

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

    @staticmethod
    def generate_start_states(num_states: int, level_seeds: list[int] | None = None) -> list[State]:
        """Generates start states for the environment.

        Args:
            num_states (int): Number of start states to generate.
            level_seeds (Optional[list[int]]): List of seeds for level generation. Defaults to
                None.

        Returns:
            list[State]: List of generated start states.
        """
        start_states: list[State] = []
        for idx in range(num_states):
            if level_seeds is not None and idx < len(level_seeds):
                state = DigitJumpState(seed=level_seeds[idx])
            else:
                state = DigitJumpState(seed=idx)
            start_states.append(state)

        return start_states

    @staticmethod
    def get_goals(states: list[State], num_steps: int | None) -> list[State]:
        """Gets the goal states for the input list of states.

        Args:
            states (list[State]): List of current states.
            num_steps (Optional[int]): Number of steps (unused in this implementation).

        Returns:
            list[State]: List of goal states.
        """
        digit_jump_states = [state for state in states if isinstance(state, DigitJumpState)]
        goal_states: list[State] = []
        for state in digit_jump_states:
            dice_puzzle_goal_state = DicePuzzle(seed=state.seed)
            dice_puzzle_goal_state.reset()
            # Check if dice_puzzle_goal_state is not None before accessing attributes
            if dice_puzzle_goal_state is not None:
                if hasattr(dice_puzzle_goal_state, "pos") and hasattr(dice_puzzle_goal_state, "end"):
                    dice_puzzle_goal_state.pos = dice_puzzle_goal_state.end
                if hasattr(dice_puzzle_goal_state, "already_solved"):
                    dice_puzzle_goal_state.already_solved = True
            goal_state = DigitJumpState(dice_puzzle=dice_puzzle_goal_state)
            goal_states.append(goal_state)

        return goal_states
