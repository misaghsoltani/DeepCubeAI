from __future__ import annotations

from copy import copy
from typing import Any

import numpy as np
from numpy import float32, uint8
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
from torch.autograd import Function as AutogradFunc
import torch.nn.functional as F

from deepcubeai.environments.environment_abstract import Environment, State
from deepcubeai.environments.puzzlegen.ice_puzzle import IcePuzzle
from deepcubeai.utils.pytorch_models import Conv2dModel, FullyConnectedModel, ResnetConv2dModel, STEThresh


class IceSliderDQN(nn.Module):
    """Deep Q-Network model for the IceSlider environment.

    Attributes:
        chan_enc (int): Number of channels for the encoder.
        enc_hw (tuple[int, int]): Height and width of the encoder.
        resnet_chan (int): Number of channels for the ResNet.
        num_resnet_blocks (int): Number of ResNet blocks.
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
        use_bias_with_norm: bool = False

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
    """Encoder model for the IceSlider environment.

    Attributes:
        chan_in (int): Number of input channels.
        chan_enc (int): Number of encoded channels.
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

        self.ste_thresh: AutogradFunc = STEThresh()

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for the encoder model.

        Args:
            states (Tensor): Current states.

        Returns:
            tuple[Tensor, Tensor]: Encoded states and rounded encoded states.
        """
        encs = self.encoder(states)
        encs_d = self.ste_thresh.apply(encs, 0.5)

        return encs, encs_d  # pyright: ignore[reportReturnType]


class Decoder(nn.Module):
    """Decoder model for the IceSlider environment.

    Attributes:
        chan_in (int): Number of input channels.
        chan_enc (int): Number of encoded channels.
        enc_hw (tuple[int, int]): Height and width of the encoded representation.
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
        """Forward pass for the decoder model.

        Args:
            encs (Tensor): Encoded states.

        Returns:
            Tensor: Decoded states.
        """
        decs = torch.reshape(encs, (encs.shape[0], self.chan_enc, self.enc_hw[0], self.enc_hw[1]))
        decs = self.decoder_conv(decs)

        return decs


class EnvModel(nn.Module):
    """Environment model for the IceSlider environment.

    Attributes:
        chan_enc (int): Number of encoded channels.
        enc_hw (tuple[int, int]): Height and width of the encoded representation.
        resnet_chan (int): Number of channels for the ResNet.
        num_resnet_blocks (int): Number of residual blocks.
        num_actions (int): Number of possible actions.
        mask_net (nn.Sequential): Sequential model containing the neural network layers.
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
        """Forward pass for the environment model.

        Args:
            states (Tensor): Current states.
            actions (Tensor): Actions to take.

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
    """Continuous environment model for the IceSlider environment.

    Attributes:
        chan_in (int): Number of input channels.
        chan_enc (int): Number of encoded channels.
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
        """Forward pass for the continuous environment model.

        Args:
            states (Tensor): Current states.
            actions (Tensor): Actions to take.

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


class IceSliderState(State):
    """State representation for the IceSlider environment.

    Attributes:
        ice_puzzle (IcePuzzle): The ice puzzle instance.
        ice_density (int): The density of ice in the puzzle.
        easy (bool): Whether the puzzle is in easy mode.
        render_style (str): The render style of the puzzle.
        min_sol_len (int): The minimum solution length.
        seed (int): The seed for random generation.
        player_x (int): The x-coordinate of the player.
        player_y (int): The y-coordinate of the player.
        hash (int): The hash value of the state.
    """

    __slots__ = [
        "ice_puzzle",
        "ice_density",
        "easy",
        "render_style",
        "min_sol_len",
        "seed",
        "player_x",
        "player_y",
        "hash",
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        self.ice_puzzle: IcePuzzle | None = None
        self.ice_density: int = 4
        self.easy: bool = True
        self.render_style: str = "human"
        self.min_sol_len: int = 8
        self.player_x: int
        self.player_y: int

        if "ice_puzzle" in kwargs:
            self.ice_puzzle = kwargs["ice_puzzle"]
            if self.ice_puzzle is not None:
                self.ice_density = self.ice_puzzle.ice_density
                self.easy = self.ice_puzzle.easy
                self.render_style = self.ice_puzzle.render_style
                self.min_sol_len = self.ice_puzzle.min_sol_len
                # Handle seed
                if hasattr(self.ice_puzzle, "seed"):
                    puzzle_seed = getattr(self.ice_puzzle, "seed", None)
                    if puzzle_seed is not None and not callable(puzzle_seed):
                        self.seed = puzzle_seed

        else:
            if "ice_density" in kwargs:
                self.ice_density = kwargs["ice_density"]

            if "easy" in kwargs:
                self.easy = kwargs["easy"]

            if "render_style" in kwargs:
                self.render_style = kwargs["render_style"]

            if "min_sol_len" in kwargs:
                self.min_sol_len = kwargs["min_sol_len"]

            if "seed" in kwargs:
                self.seed = kwargs["seed"]

            self.ice_puzzle = IcePuzzle(
                ice_density=self.ice_density,
                easy=self.easy,
                render_style=self.render_style,
                min_sol_len=self.min_sol_len,
                seed=self.seed,
            )
            self.ice_puzzle.reset()

        if self.ice_puzzle is not None:
            if hasattr(self.ice_puzzle, "pos") and self.ice_puzzle.pos is not None:
                self.player_x = self.ice_puzzle.pos[0]
                self.player_y = self.ice_puzzle.pos[1]
            else:
                self.player_x = 0
                self.player_y = 0
        else:
            self.player_x = 0
            self.player_y = 0
        self.hash: int | None = None

    def __hash__(self) -> int:
        """Return hash value for the state."""
        if self.hash is None:
            if self.ice_puzzle is not None and hasattr(self.ice_puzzle, "grid"):
                grid_str = str(getattr(self.ice_puzzle, "grid", ""))
                start_val = getattr(self.ice_puzzle, "start", (0, 0))
                end_val = getattr(self.ice_puzzle, "end", (0, 0))
                self.hash = hash((grid_str, start_val, end_val, self.player_x, self.player_y))
            else:
                self.hash = hash((self.player_x, self.player_y))
        return self.hash

    def __eq__(self, other: object) -> bool:
        """Check equality with another state."""
        if not isinstance(other, IceSliderState):
            return False
        if (
            self.ice_puzzle is not None
            and other.ice_puzzle is not None
            and hasattr(self.ice_puzzle, "grid")
            and hasattr(other.ice_puzzle, "grid")
        ):
            try:
                grid1 = getattr(self.ice_puzzle, "grid", None)
                grid2 = getattr(other.ice_puzzle, "grid", None)
                start1 = getattr(self.ice_puzzle, "start", None)
                start2 = getattr(other.ice_puzzle, "start", None)
                end1 = getattr(self.ice_puzzle, "end", None)
                end2 = getattr(other.ice_puzzle, "end", None)

                return (
                    (grid1 is not None and grid2 is not None and np.array_equal(grid1, grid2))
                    and start1 == start2
                    and end1 == end2
                    and self.player_x == other.player_x
                    and self.player_y == other.player_y
                )
            except Exception:
                return self.player_x == other.player_x and self.player_y == other.player_y
        else:
            return self.player_x == other.player_x and self.player_y == other.player_y

    def move(self, action: int) -> tuple[float, bool, float, IceSliderState]:
        """Perform a move in the environment.

        Args:
            action (int): The action to take.

        Returns:
            tuple[float, bool, float, IceSliderState]: A tuple containing the reward, whether the
                move is done, the move cost, and the next state.
        """
        if self.ice_puzzle is None:
            raise ValueError("ice_puzzle is not initialized")
        next_ice_puzzle = copy(self.ice_puzzle)
        if next_ice_puzzle is None:
            raise ValueError("Failed to copy ice_puzzle")

        raw_reward: float = 0.0
        raw_done: bool = False

        try:
            rr, rd, _ = next_ice_puzzle.step(int(action))
            raw_reward, raw_done = float(rr), bool(rd)
        except Exception:
            # Fallback to default values if step fails
            raw_reward, raw_done = 0.0, False

        # Normalize types
        reward_float = float(raw_reward)
        done = bool(raw_done)

        next_state = IceSliderState(**{"ice_puzzle": next_ice_puzzle})
        # We assume all actions have a uniform cost of 1.0
        move_cost = 1.0
        return reward_float, done, move_cost, next_state

    def get_solution(self) -> list[int]:
        """Get the solution to the puzzle.

        Returns:
            list[int]: The list of actions to solve the puzzle.
        """
        if self.ice_puzzle is None or not hasattr(self.ice_puzzle, "solution"):
            return []
        solution = getattr(self.ice_puzzle, "solution", None)
        return solution if solution is not None else []

    def get_opt_path_len(self) -> int:
        """Get the length of the optimal path.

        Returns:
            int: The length of the optimal path.
        """
        if self.ice_puzzle is None or not hasattr(self.ice_puzzle, "solution"):
            return 0
        solution = getattr(self.ice_puzzle, "solution", None)
        return len(solution) if solution is not None else 0

    def render_image(self) -> NDArray[uint8]:
        """Render the puzzle as an image.

        Returns:
            NDArray: The rendered image of the puzzle.
        """
        if self.ice_puzzle is None:
            return np.zeros((64, 64, 3), dtype=uint8)
        rendered = self.ice_puzzle.render()
        # IcePuzzle.render() returns an image; cast to uint8 for consistency
        return rendered.astype(uint8)

    def _set_pos(self, x: int, y: int) -> bool:
        """Set the position of the player.

        Args:
            x (int): The x-coordinate.
            y (int): The y-coordinate.

        Returns:
            bool: True if the position is set successfully, False otherwise (if there is a rock in
                that cell).
        """
        # the agent only can be in the cells that are not rock
        if self.ice_puzzle is None:
            return False

        grid = self._get_grid()
        if grid[x][y] != 0:
            try:
                if hasattr(self.ice_puzzle, "pos") and self.ice_puzzle.pos is not None:
                    self.ice_puzzle.pos = (x, y)
                self.player_x = x
                self.player_y = y
                return True
            except Exception:
                return False

        return False

    def _get_grid(self) -> NDArray[uint8]:
        """Get the grid representation of the puzzle.

        Returns:
            NDArray: The grid representation of the puzzle.
        """
        if self.ice_puzzle is None:
            return np.zeros((8, 8), dtype=uint8)
        grid = getattr(self.ice_puzzle, "grid", None)
        return grid if grid is not None else np.zeros((8, 8), dtype=uint8)


class IceSliderEnvironment(Environment):
    """IceSlider environment.

    Attributes:
        moves (list[str]): List of possible moves.
        shape (int): Shape of the environment.
        num_moves (int): Number of possible moves.
        chan_enc (int): Number of encoded channels.
        enc_hw (tuple[int, int]): Height and width of the encoded representation.
    """

    moves: list[str] = ["UP", "RIGHT", "LEFT", "DOWN", "NO-OP"]

    # directions = {0: 'Up', 1: 'Right', 2: 'Left', 3: 'Down', 4: 'No-op'}

    def __init__(self, shape: int = 64) -> None:
        super().__init__()
        self.shape: int = shape
        self.num_moves: int = len(self.moves)
        self.chan_enc: int = 3
        enc_h: int = 8
        enc_w: int = 8
        self.enc_hw: tuple[int, int] = (enc_h, enc_w)

    @property
    def env_name(self) -> str:
        """Gets the name of the environment.

        Returns:
            str: The name of the environment, "iceslider".
        """
        return "iceslider"

    @property
    def num_actions_max(self) -> int:
        """Get the maximum number of actions.

        Returns:
            int: The maximum number of actions.
        """
        return self.num_moves

    @staticmethod
    def next_state(states: list[State], actions: list[int]) -> tuple[list[State], list[float]]:
        """Get the next state and transition cost given the current state and action.

        Args:
            states (list[State]): List of current states.
            actions (list[int]): List of actions to take.

        Returns:
            tuple[list[State], list[float]]: The next states and transition costs.
        """
        next_states: list[State] = []
        transition_costs: list[float] = []
        for state, action in zip(states, actions, strict=False):
            if isinstance(state, IceSliderState):
                _, _, move_cost, next_state = state.move(action)
                next_states.append(next_state)
                transition_costs.append(move_cost)

        return next_states, transition_costs

    def rand_action(self, states: list[State]) -> list[int]:
        """Get random actions that could be taken in each state.

        Args:
            states (list[State]): List of current states.

        Returns:
            list[int]: List of random actions.
        """
        return list(np.random.randint(0, self.num_actions_max, size=len(states)))

    @staticmethod
    def is_solved(states: list[State], states_goal: list[State]) -> NDArray[np.bool_]:
        """Check if the states are solved.

        Args:
            states (list[State]): List of current states.
            states_goal (list[State]): List of goal states.

        Returns:
            NDArray: Boolean array indicating whether each state is solved.
        """
        ice_states = [state for state in states if isinstance(state, IceSliderState)]
        ice_goal_states = [state for state in states_goal if isinstance(state, IceSliderState)]

        if not ice_states or not ice_goal_states:
            return np.array([False] * len(states), dtype=np.bool_)

        states_np = np.stack([(state.player_x, state.player_y) for state in ice_states], axis=0)
        goal_states_np = np.stack(
            [(goal_state.player_x, goal_state.player_y) for goal_state in ice_goal_states], axis=0
        )

        return np.all(np.equal(states_np, goal_states_np), axis=1)

    def state_to_real(self, states: list[State]) -> NDArray[float32]:
        """Convert states to real-world observations.

        Args:
            states (list[State]): List of current states.

        Returns:
            NDArray: Real-world observations.
        """
        states_real: NDArray[float32] = np.zeros((len(states), self.shape, self.shape, 3), dtype=float32)
        for state_idx, state in enumerate(states):
            if isinstance(state, IceSliderState):
                states_real[state_idx] = state.render_image() / 255.0
            else:
                states_real[state_idx] = np.zeros((self.shape, self.shape, 3), dtype=float32)

        states_real = states_real.transpose([0, 3, 1, 2])

        return states_real

    def get_dqn(self) -> nn.Module:
        """Get the DQN model for the environment.

        Returns:
            nn.Module: The DQN model.
        """
        resnet_chan: int = 7 * self.chan_enc * 2
        num_resnet_blocks: int = 4
        return IceSliderDQN(self.chan_enc, self.enc_hw, resnet_chan, num_resnet_blocks, self.num_actions_max, True)

    def get_env_nnet(self) -> nn.Module:
        """Get the environment neural network model.

        Returns:
            nn.Module: The environment neural network model.
        """
        resnet_chan: int = self.chan_enc + self.num_actions_max
        num_resnet_blocks: int = 4
        return EnvModel(self.chan_enc, self.enc_hw, resnet_chan, num_resnet_blocks, self.num_actions_max)

    def get_env_nnet_cont(self) -> nn.Module:
        """Get the continuous environment neural network model.

        Returns:
            nn.Module: The continuous environment neural network model.
        """
        resnet_chan: int = self.chan_enc + self.num_actions_max
        num_resnet_blocks: int = 4
        return EnvModelContinuous(3, self.chan_enc, resnet_chan, num_resnet_blocks, self.num_actions_max)

    def get_encoder(self) -> nn.Module:
        """Get the encoder model.

        Returns:
            nn.Module: The encoder model.
        """
        return Encoder(3, self.chan_enc)

    def get_decoder(self) -> nn.Module:
        """Get the decoder model.

        Returns:
            nn.Module: The decoder model.
        """
        return Decoder(3, self.chan_enc, self.enc_hw)

    @staticmethod
    def generate_start_states(num_states: int, level_seeds: list[int] | None = None) -> list[State]:
        """Generate start states for the environment.

        Args:
            num_states (int): Number of start states to generate.
            level_seeds (Optional[list[int]], optional): List of seeds for level generation, by
                default None.

        Returns:
            list[State]: List of generated start states.
        """
        start_states: list[State] = []
        for idx in range(num_states):
            seed_idx = None if level_seeds is None else level_seeds[idx]
            state = IceSliderState(**{"seed": seed_idx}) if seed_idx is not None else IceSliderState()
            start_states.append(state)

        return start_states

    @staticmethod
    def get_goals(states: list[State], num_steps: int | None) -> list[State]:
        """Get the goal states for the input list of states.

        Args:
            states (list[State]): List of current states.
            num_steps (Optional[int]): Unused for this environment.

        Returns:
            list[State]: List of goal states.
        """
        goal_states: list[State] = []
        for state in states:
            ice_puzzle_goal_state = IcePuzzle(seed=state.seed)
            ice_puzzle_goal_state.reset()
            ice_puzzle_goal_state.pos = ice_puzzle_goal_state.end
            ice_puzzle_goal_state.already_solved = True
            goal_state = IceSliderState(**{"ice_puzzle": ice_puzzle_goal_state})
            goal_states.append(goal_state)

        return goal_states
