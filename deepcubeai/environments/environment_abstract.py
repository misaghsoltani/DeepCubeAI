from abc import ABC, abstractmethod
from inspect import signature
from typing import List, Optional, Tuple

import numpy as np
from torch import nn

from deepcubeai.utils.decorators import enforce_init_defaults, optional_abstract_method


class State(ABC):

    def __init__(self):
        self.seed = None

    @optional_abstract_method
    def get_opt_path_len(self) -> int:
        """Get the length of the optimal path.

        Returns:
            int: The length of the optimal path.
        """

    @optional_abstract_method
    def get_solution(self) -> List[int]:
        """Get the list of actions to be taken to get to the goal.

        Returns:
            List[int]: The list of actions to be taken to get to the goal.
        """


@enforce_init_defaults
class Environment(ABC):

    def __init__(self):
        self.dtype = float
        self.fixed_actions: bool = True

    @abstractmethod
    def get_env_name(self) -> str:
        """Gets the name of the environment.

        Returns:
            str: The name of the environment.
        """

    @property
    @abstractmethod
    def num_actions_max(self):
        pass

    @abstractmethod
    def next_state(self, states: List[State],
                   actions: List[int]) -> Tuple[List[State], List[float]]:
        """Get the next state and transition cost given the current state and action.

        Args:
            states (List[State]): List of states.
            actions (List[int]): Actions to take.

        Returns:
            Tuple[List[State], List[float]]: Next states, transition costs. Input states may
                be modified!
        """

    @abstractmethod
    def rand_action(self, states: List[State]) -> List[int]:
        """Get random actions that could be taken in each state.

        Args:
            states (List[State]): List of states.

        Returns:
            List[int]: List of random actions.
        """

    @abstractmethod
    def is_solved(self, states: List[State], states_goal: List[State]) -> np.array:
        """Returns whether or not state is solved.

        Args:
            states (List[State]): List of states.
            states_goal (List[State]): List of goal states.

        Returns:
            np.array: Boolean numpy array where the element at index i corresponds to whether or
                not the state at index i is solved.
        """

    @abstractmethod
    def state_to_real(self, states: List[State]) -> np.ndarray:
        """State to real-world observation.

        Args:
            states (List[State]): List of states.

        Returns:
            np.ndarray: A numpy array.
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
    def generate_start_states(self,
                              num_states: int,
                              level_seeds: Optional[List[int]] = None) -> List[State]:
        pass

    @optional_abstract_method
    def get_goals(self, states: List[State], num_steps: Optional[int]) -> List[State]:
        """Get the goal states for the input list of states.

        Args:
            states (List[State]): List of states.
            num_steps (Optional[int]): Number of random steps to be taken to specify the resulting
                state as a goal state. This may or may not be used in different environments.

        Returns:
            List[State]: List of goal states.
        """

    def generate_episodes(
        self,
        num_steps_l: List[float],
        start_level_seed: Optional[int] = -1,
        num_levels: Optional[int] = -1
    ) -> Tuple[List[State], List[State], List[List[State]], List[List[int]]]:
        """Generate episodes based on the given parameters.

        Args:
            num_steps_l (List[float]): List of number of steps for each trajectory.
            start_level_seed (Optional[int], optional): Starting seed for level generation,
                defaults to -1.
            num_levels (Optional[int], optional): Number of levels to generate, defaults to -1.

        Returns:
            Tuple[List[State], List[State], List[List[State]], List[List[int]]]: Tuple containing
                start states, goal states, trajectories, and action trajectories.
        """

        num_trajs: int = len(num_steps_l)

        # Check if the implemented method 'generate_start_states()' accepts 'level_seeds' as
        # an argument
        has_arg: bool = "level_seeds" in signature(self.generate_start_states).parameters

        # Initialize
        states: List[State]
        if has_arg:
            # Calculating the seeds
            seeds_lst: List[int] = None
            if (num_levels > 0) or (start_level_seed > -1):
                if num_levels < 1:
                    num_levels = num_trajs

                elif start_level_seed < 0:
                    start_level_seed = np.random.randint(0, 1000000)

                trajs_per_level = num_trajs // num_levels
                extra_trajs = num_trajs % num_levels
                levels = np.arange(start_level_seed, start_level_seed + num_levels)
                seeds_np = np.concatenate((np.tile(levels, trajs_per_level), levels[:extra_trajs]))
                np.random.shuffle(seeds_np)
                seeds_lst = seeds_np.tolist()

            states = self.generate_start_states(num_trajs, level_seeds=seeds_lst)

        else:
            states = self.generate_start_states(num_trajs)

        states_walk: List[State] = [state for state in states]

        # Num steps
        num_steps: np.array = np.array(num_steps_l)
        num_moves_curr: np.array = np.zeros(len(states))

        # random walk
        trajs: List[List[State]] = [[state] for state in states]
        action_trajs: List[List[int]] = [[] for _ in range(len(states))]

        moves_lt = num_moves_curr < num_steps
        while np.any(moves_lt):
            idxs: np.ndarray = np.where(moves_lt)[0]
            states_to_move = [states_walk[idx] for idx in idxs]

            actions: List[int] = self.rand_action(states_to_move)
            states_moved, _ = self.next_state(states_to_move, actions)

            for move_idx, idx in enumerate(idxs):
                trajs[idx].append(states_moved[move_idx])
                action_trajs[idx].append(actions[move_idx])
                states_walk[idx] = states_moved[move_idx]

            num_moves_curr[idxs] = num_moves_curr[idxs] + 1

            moves_lt[idxs] = num_moves_curr[idxs] < num_steps[idxs]

        # get state goal pairs
        states_start: List[State] = [traj[0] for traj in trajs]
        states_goal: List[State] = [traj[-1] for traj in trajs]

        return states_start, states_goal, trajs, action_trajs
