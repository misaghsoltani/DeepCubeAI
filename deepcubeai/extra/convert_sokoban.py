import pickle
from typing import Dict, List

import numpy as np

from deepcubeai.environments.sokoban import SokobanState


def main():
    """Converts old Sokoban states to new format and save them.

    This function initializes the environment, loads old states from a pickle file,
    converts them to the new format, and saves the converted states and goals to a new pickle file.
    """

    wall_idxs_old: np.array = np.arange(0, 100, 1)
    goal_idxs_old: np.array = np.arange(100, 200, 1)
    box_idxs_old: np.array = np.arange(200, 300, 1)
    agent_idxs_old: np.array = np.arange(300, 400, 1)

    with open("data/sokoban/test/states_old.pkl", "rb") as file:
        states_old = pickle.load(file)["states"]

    states: List[SokobanState] = []
    state_goals: List[SokobanState] = []

    for state_old in states_old:
        wall_mask = state_old[wall_idxs_old].reshape(10, 10).astype(bool)
        goal_mask = state_old[goal_idxs_old].reshape(10, 10).astype(bool)
        box_mask = state_old[box_idxs_old].reshape(10, 10).astype(bool)
        agent_idx = np.where(state_old[agent_idxs_old].reshape(10, 10))
        agent_idx = np.array([agent_idx[0][0], agent_idx[1][0]], dtype=int)

        state_start = SokobanState(agent_idx, box_mask, wall_mask)
        state_goal = SokobanState(agent_idx, goal_mask, wall_mask)

        states.append(state_start)
        state_goals.append(state_goal)

    data: Dict = {"states": states, "state_goals": state_goals}

    with open("data/sokoban/test/data.pkl", "wb") as file:
        pickle.dump(data, file, protocol=-1)


if __name__ == "__main__":
    main()
