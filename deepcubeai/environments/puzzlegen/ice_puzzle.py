from __future__ import annotations

import os

import cv2
import networkx as nx
from networkx import DiGraph, MultiDiGraph
import numpy as np
from numpy import uint8
from numpy.typing import NDArray

from deepcubeai.environments.puzzlegen.base import PuzzleEnv
from deepcubeai.utils.misc_utils import imread_cv2


class IcePuzzle(PuzzleEnv):
    """IceSlider environment."""

    # For static analyzers
    rock_rgb: NDArray[uint8]
    ice_rgb: NDArray[uint8]
    player_rgb: NDArray[uint8]
    pos: tuple[int, int] | None = None
    end: tuple[int, int] | None = None
    solution: list[int] = []
    already_solved: bool = False
    goal_rgb: NDArray[uint8]

    def __init__(
        self,
        ice_density: int = 4,
        easy: bool = True,
        render_style: str = "human",
        min_sol_len: int = 8,
        seed: int | None = None,
        size: int = 8,
        max_tries: int = 1_000_000,
    ) -> None:
        super().__init__(seed=seed, size=size, render_style=render_style, min_sol_len=min_sol_len, max_tries=max_tries)
        self.easy: bool = easy
        self.ice_density: int = ice_density
        self.grid: NDArray[uint8]

        if self.render_style == "human":
            self.rock_rgb = imread_cv2(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/rock.png"), dtype=uint8
            )
            self.ice_rgb = imread_cv2(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/ice.png"), dtype=uint8
            )
            self.player_rgb = imread_cv2(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/player.png"), dtype=uint8
            )
            # self.goal_rgb = imread_cv2(
            #     os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/goal.png"), dtype=uint8
            # )
            self.goal_rgb = imread_cv2(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/ice.png"), dtype=uint8
            )

        elif self.render_style == "grid_world":
            self.rock_rgb = np.array([[[255, 0, 0]]])
            self.ice_rgb = np.array([[[255, 255, 255]]])
            self.player_rgb = np.array([[[0, 255, 0]]])
            # self.goal_rgb = np.array([[[0, 0, 255]]])
            self.goal_rgb = np.array([[[255, 255, 255]]])

        else:
            raise Exception("Unknown rendering mode.")

    def _can_go(self, r: int, c: int, grid: list[list[int]] | NDArray[uint8]) -> bool:
        """Check if the player can go to the specified position.

        Args:
            r (int): Row index.
            c (int): Column index.
            grid (list[list[int]] | NDArray[uint8]): The grid to check.

        Returns:
            bool: True if the position is valid, False otherwise.
        """
        is_valid = (0 <= r < self.size) and (0 <= c < self.size)
        return is_valid and bool(grid[r][c])

    def _slide(self, r: int, c: int, grid: list[list[int]] | NDArray[uint8], d: int) -> tuple[int, int]:
        """Slide the player in the specified direction until hitting an obstacle.

        Args:
            r (int): Row index.
            c (int): Column index.
            grid (list[list[int]] | NDArray[uint8]): The grid to slide on.
            d (int): Direction index (0: up, 1: right, 2: down, 3: left).

        Returns:
            tuple[int, int]: The new position after sliding.
        """
        if not self._can_go(r + self.y[d], c + self.x[d], grid):
            return r, c

        return self._slide(r + self.y[d], c + self.x[d], grid, d)

    def _create_level(self) -> None:
        """Create a new level for the environment."""
        node_id: dict[tuple[int, int], int]
        edges: list[tuple[int, int]]
        q: list[tuple[tuple[int, int], list[int]]]
        z: dict[tuple[int, int], int]
        grid: list[list[int]]
        for _ in range(self.max_tries):
            start = (0, self.rng.randint(0, self.size - 1))
            end = (self.size - 1, self.rng.randint(0, self.size - 1))
            grid = [[self.rng.randint(0, self.ice_density) for _ in range(self.size)] for _ in range(self.size)]
            grid[0][start[1]] = 1
            grid[self.size - 1][end[1]] = 1
            q = [(start, [])]
            n = start
            z = {}
            while q and n != end:
                n, path = q.pop()
                for d in range(4):
                    n_next = self._slide(*n, grid, d)
                    if self._can_go(n[0] + self.y[d], n[1] + self.x[d], grid) and n_next not in z:
                        q[:0] = [(n_next, path + [d])]
                        z[n_next] = 1

            if (n == end) * len(path) > self.min_sol_len:
                if self.easy:
                    break

                else:
                    # The shortest solution is longer than required, now we check that you can get
                    # stuck a strongly connected component is reachable from the start, but not
                    # connected to the end
                    g: MultiDiGraph[int] = nx.MultiDiGraph()
                    count = 0
                    node_id = {}
                    for i in range(self.size):
                        for j in range(self.size):
                            if grid[i][j]:
                                node_id[(i, j)] = count
                                count += 1
                    edges = []
                    for pos in node_id:
                        for d in range(4):
                            edges.append((node_id[pos], node_id[self._slide(*pos, grid, d)]))

                    g.add_nodes_from(np.arange(count))
                    g.add_edges_from(edges)
                    c: DiGraph[int] = nx.condensation(g)

                    if any(
                        nx.has_path(c, c.graph["mapping"][node_id[start]], n)
                        and not nx.has_path(c, n, c.graph["mapping"][node_id[end]])
                        for n in c.nodes
                    ):
                        break

        self.grid = (np.array(grid) != 0).astype(uint8)
        self.pos = start
        self.end = end
        self.solution = path

    def _reset(self) -> NDArray[uint8]:
        """Reset the environment to initial state."""
        self._create_level()
        return self.render()

    def _step(self, a: int) -> tuple[float, bool, dict[str, int]]:
        """Take a step in the environment with action a.

        Args:
            a (int): Action to take.

        Returns:
            tuple[float, bool, dict[str, int]]: Reward, done flag, and info dictionary.
        """
        if a in self.action_space and a < 4:
            assert self.pos is not None
            r, c = self.pos
            self.pos = self._slide(r, c, self.grid, a)

        reward, done = (10.0, True) if (self.pos == self.end and not self.already_solved) else (0, False)
        self.already_solved = True if self.pos == self.end else self.already_solved

        # return self.render(), reward, done, self.labels()
        return reward, done, self.labels()

    def _get_image(self) -> NDArray[uint8]:
        """Render the environment to an image."""
        rgb = np.concatenate(
            [
                np.concatenate(
                    [
                        self.rock_rgb
                        if not el
                        else (
                            self.player_rgb
                            if self.pos == (i, j)
                            else (self.goal_rgb if self.end == (i, j) else self.ice_rgb)
                        )
                        for j, el in enumerate(row)
                    ],
                    axis=1,
                )
                for i, row in enumerate(self.grid)
            ],
            axis=0,
        )
        rescaled = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_NEAREST)
        render = np.clip(rescaled, 0, 255)

        return render.astype(uint8)
