from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature
from typing import Any

import numpy as np
from numpy import float32, intp
from numpy.typing import NDArray
from torch import nn

from deepcubeai.utils.decorators import enforce_init_defaults, optional_abstract_method


class State(ABC):
    """Abstract base class for environment states.

    Attributes:
        seed: Optional seed value for random number generation.
    """

    def __init__(self) -> None:
        self.seed: int | None = None

    @abstractmethod
    def __hash__(self) -> int:
        """Hash method for the state."""
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Equality method for the state."""
        pass

    @optional_abstract_method
    def get_opt_path_len(self) -> int:
        """Get the length of the optimal path.

        Returns:
            int: The length of the optimal path.
        """
        raise NotImplementedError()

    @optional_abstract_method
    def get_solution(self) -> list[int]:
        """Get the list of actions to be taken to get to the goal.

        Returns:
            list[int]: The list of actions to be taken to get to the goal.
        """
        raise NotImplementedError()


@enforce_init_defaults
class Environment(ABC):
    """Abstract base class for environments.

    Attributes:
        dtype: Data type for numerical computations.
        fixed_actions: Whether the environment has a fixed set of actions.
    """

    def __init__(self) -> None:
        # dtype is a type-like field used by environments (e.g., numpy scalar types)
        self.dtype: type[Any] = float
        self.fixed_actions: bool = True

    @property
    @abstractmethod
    def env_name(self) -> str:
        """Gets the name of the environment.

        Returns:
            str: The name of the environment.
        """
        pass

    @property
    @abstractmethod
    def num_actions_max(self) -> int:
        """Get the maximum number of actions.

        Returns:
            int: Maximum number of actions.
        """
        pass

    @abstractmethod
    def next_state(self, states: list[State], actions: list[int]) -> tuple[list[State], list[float]]:
        """Get the next state and transition cost given the current state and action.

        Args:
            states (list[State]): List of states.
            actions (list[int]): Actions to take.

        Returns:
            tuple[list[State], list[float]]: Next states, transition costs. Input states may
                be modified!
        """

    @abstractmethod
    def rand_action(self, states: list[State]) -> list[int]:
        """Get random actions that could be taken in each state.

        Args:
            states (list[State]): List of states.

        Returns:
            list[int]: List of random actions.
        """

    @staticmethod
    @abstractmethod
    def is_solved(states: list[State], states_goal: list[State]) -> NDArray[np.bool_]:
        """Returns whether or not state is solved.

        Args:
            states (list[State]): List of states.
            states_goal (list[State]): List of goal states.

        Returns:
            NDArray: Boolean numpy array where the element at index i corresponds to whether or
                not the state at index i is solved.
        """

    @abstractmethod
    def state_to_real(self, states: list[State]) -> NDArray[float32]:
        """State to real-world observation.

        Args:
            states (list[State]): List of states.

        Returns:
            NDArray: A numpy array.
        """

    @abstractmethod
    def get_dqn(self) -> nn.Module:
        """Get the neural network model for the dqn.

        Returns:
            nn.Module: Neural network model.
        """

    @abstractmethod
    def get_env_nnet(self) -> nn.Module:
        """Get the neural network model for the environment.

        Returns:
            nn.Module: Neural network model.
        """

    @abstractmethod
    def get_env_nnet_cont(self) -> nn.Module:
        """Get the neural network model for the environment for the continuous setting.

        Returns:
            nn.Module: Neural network model.
        """

    @abstractmethod
    def get_encoder(self) -> nn.Module:
        """Get encoder.

        Returns:
            nn.Module: Neural network model.
        """

    @abstractmethod
    def get_decoder(self) -> nn.Module:
        """Get decoder.

        Returns:
            nn.Module: Neural network model.
        """

    @abstractmethod
    def generate_start_states(self, num_states: int, level_seeds: list[int] | None = None) -> list[State]:
        """Generate start states for the environment.

        Args:
            num_states: Number of states to generate.
            level_seeds: Optional list of seeds for level generation.

        Returns:
            List of generated start states.
        """
        pass

    @optional_abstract_method
    @staticmethod
    def get_goals(states: list[State], num_steps: int | None) -> list[State]:
        """Get the goal states for the input list of states.

        Args:
            states (list[State]): List of states.
            num_steps (Optional[int]): Number of random steps to be taken to specify the resulting
                state as a goal state. This may or may not be used in different environments.

        Returns:
            list[State]: List of goal states.
        """
        raise NotImplementedError()

    def generate_episodes(
        self, num_steps_l: list[float], start_level_seed: int | None = -1, num_levels: int | None = -1
    ) -> tuple[list[State], list[State], list[list[State]], list[list[int]]]:
        """Generate episodes based on the given parameters.

        Args:
            num_steps_l (list[float]): List of number of steps for each trajectory.
            start_level_seed (Optional[int], optional): Starting seed for level generation,
                defaults to -1.
            num_levels (Optional[int], optional): Number of levels to generate, defaults to -1.

        Returns:
            tuple[list[State], list[State], list[list[State]], list[list[int]]]: Tuple containing
                start states, goal states, trajectories, and action trajectories.
        """
        num_trajs: int = len(num_steps_l)

        # Check if the implemented method 'generate_start_states()' accepts 'level_seeds' as an argument
        has_arg: bool = "level_seeds" in signature(self.generate_start_states).parameters

        # Initialize
        states: list[State]
        if has_arg:
            # Calculating the seeds
            seeds_lst: list[int] | None = None
            if (num_levels is not None and num_levels > 0) or (start_level_seed is not None and start_level_seed > -1):
                if num_levels is None or num_levels < 1:
                    num_levels = num_trajs

                if start_level_seed is None or start_level_seed < 0:
                    start_level_seed = np.random.randint(0, 1000000)

                trajs_per_level = num_trajs // num_levels
                extra_trajs = num_trajs % num_levels
                levels: NDArray[intp] = np.arange(start_level_seed, start_level_seed + num_levels, dtype=intp)
                seeds_np = np.concatenate((np.tile(levels, trajs_per_level), levels[:extra_trajs]))
                np.random.shuffle(seeds_np)
                seeds_lst = seeds_np.tolist()

            states = self.generate_start_states(num_trajs, level_seeds=seeds_lst)

        else:
            states = self.generate_start_states(num_trajs)

        states_walk: list[State] = list(states)

        # Num steps
        num_steps: NDArray[float32] = np.array(num_steps_l, dtype=float32)
        num_moves_curr: NDArray[float32] = np.zeros(len(states), dtype=float32)

        # Random walk
        trajs: list[list[State]] = [[state] for state in states]
        action_trajs: list[list[int]] = [[] for _ in range(len(states))]

        moves_lt = num_moves_curr < num_steps
        while np.any(moves_lt):
            idxs: NDArray[intp] = np.where(moves_lt)[0]
            states_to_move = [states_walk[idx] for idx in idxs]

            actions: list[int] = self.rand_action(states_to_move)
            states_moved, _ = self.next_state(states_to_move, actions)

            for move_idx, idx in enumerate(idxs):
                trajs[idx].append(states_moved[move_idx])
                action_trajs[idx].append(actions[move_idx])
                states_walk[idx] = states_moved[move_idx]

            num_moves_curr[idxs] += 1

            moves_lt[idxs] = num_moves_curr[idxs] < num_steps[idxs]

        # Get state goal pairs
        states_start: list[State] = [traj[0] for traj in trajs]
        states_goal: list[State] = [traj[-1] for traj in trajs]

        return states_start, states_goal, trajs, action_trajs
