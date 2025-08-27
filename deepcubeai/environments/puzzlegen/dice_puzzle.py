from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
from numpy import uint8
from numpy.typing import NDArray

from deepcubeai.environments.puzzlegen.base import PuzzleEnv


class DicePuzzle(PuzzleEnv):
    """DigitJump environment."""

    palette: list[NDArray[Any]]
    grid: NDArray[uint8] | None
    pos: tuple[int, int] | None
    start: tuple[int, int] | None
    end: tuple[int, int] | None
    solution: list[int] | None
    already_solved: bool | None
    targets: NDArray[Any] | None
    images: NDArray[Any] | None
    texture: list[NDArray[Any]]
    player_rgb: NDArray[Any]
    non_player_rgb: NDArray[Any]

    def __init__(
        self,
        render_style: str = "mnist",
        seed: int | None = None,
        size: int = 8,
        min_sol_len: int = 1,
        max_tries: int = 1_000_000,
    ) -> None:
        super().__init__(seed=seed, size=size, render_style=render_style, min_sol_len=min_sol_len, max_tries=max_tries)

    def _can_go(self, r: int, c: int) -> bool:
        """Check if the player can go to the specified position.

        Args:
            r (int): Row index.
            c (int): Column index.

        Returns:
            bool: True if the position is valid, False otherwise.
        """
        return (0 <= r < self.size) and (0 <= c < self.size)

    def _move(self, r: int, c: int, d: int, dist: int) -> tuple[int, int]:
        """Move the player in the specified direction by the specified distance.

        Args:
            r (int): Row index.
            c (int): Column index.
            d (int): Direction index (0: up, 1: right, 2: down, 3: left).
            dist (int): Distance to move.

        Returns:
            tuple[int, int]: New position after the move.
        """
        new_pos = (r + self.y[d] * dist, c + self.x[d] * dist)
        return new_pos if self._can_go(*new_pos) else (r, c)

    def _reset(self) -> NDArray[uint8]:
        """Reset the environment to initial state."""
        self._create_level()
        return self.render()

    def _create_level(self) -> None:
        """Create a new level for the environment."""
        for _ in range(self.max_tries):
            start = (0, 0)
            end = (self.size - 1, self.size - 1)
            grid = [[self.rng.randint(1, min(6, self.size - 1)) for _ in range(self.size)] for _ in range(self.size)]
            q: list[tuple[tuple[int, int], list[int]]] = [(start, [])]
            n = start
            z: dict[tuple[int, int], int] = {}
            while q and n != end:
                n, path = q.pop()
                for d in range(4):
                    n_next = self._move(*n, d, grid[n[0]][n[1]])
                    if n_next not in z:
                        q[:0] = [(n_next, path + [d])]
                        z[n_next] = 1
            if n == end:
                break
        self.grid = np.array(grid).astype(uint8)
        self.pos = start
        self.end = end
        self.solution = path

        if self.render_style == "mnist":
            data = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/mnist.csv"), delimiter=",")
            self.targets = data[:, -1]
            self.images = 1 - data[:, :-1].reshape(-1, 8, 8, 1).astype(float) / 16
            self.texture = [self.images[list(np.where(self.targets == i)[0])[0]] for i in range(7)][1:]
            self.player_rgb = (
                np.stack([0.923 * np.ones((8, 8)), 0.386 * np.ones((8, 8)), 0.209 * np.ones((8, 8))], axis=-1).astype(
                    float
                )
                * 255
            )
            self.non_player_rgb = (
                np.stack([0.56 * np.ones((8, 8)), 0.692 * np.ones((8, 8)), 0.195 * np.ones((8, 8))], axis=-1).astype(
                    float
                )
                * 255
            )
        elif self.render_style == "grid_world":
            self.player_rgb = (
                np.pad(np.ones((6, 6)), ((1, 1), (1, 1)), mode="constant", constant_values=0)
                .reshape((8, 8, 1))
                .astype(float)
            )
            # Define the RGB palette as a list of color triplets, then expand each into an 8x8x3 tile.
            palette_colors: list[list[int]] = [
                [132, 94, 194],
                [214, 93, 177],
                [255, 111, 145],
                [255, 150, 113],
                [255, 199, 95],
                [249, 248, 113],
            ]
            self.palette = [
                np.stack([np.stack([np.array(color, dtype=float)] * 8, axis=0)] * 8, axis=0).astype(float)
                for color in palette_colors
            ]
        elif self.render_style == "dice":
            data = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/faces.csv"), delimiter=",")
            images = data.reshape((-1, 8, 8, 1)).astype(float)
            self.texture = list(images)
            self.player_rgb = np.ones((8, 8, 3)).astype(float) * 255
            palette_lst: list[list[int]] = [
                [132, 94, 194],
                [214, 93, 177],
                [255, 111, 145],
                [255, 150, 113],
                [255, 199, 95],
                [249, 248, 113],
            ]
            self.palette = [
                np.stack([np.stack([np.array(p)] * 8, axis=0, dtype=uint8)] * 8, axis=0, dtype=uint8)
                for p in palette_lst
            ]
        elif self.render_style == "beta":
            # Gradient palette tiles
            self.palette = [
                np.stack([np.ones((8, 8)), np.zeros((8, 8)), np.zeros((8, 8))], axis=-1) * float(i) * (255 / 6)
                for i in range(7)
            ][1:]
            self.player_rgb = (
                np.stack(
                    [
                        np.zeros((8, 8)),
                        np.pad(np.ones((4, 4)), ((2, 2), (2, 2)), mode="constant", constant_values=0),
                        np.zeros((8, 8)),
                    ],
                    axis=-1,
                ).astype(float)
                * 255
            )
        else:
            raise Exception("Unknown rendering mode.")

    def _step(self, a: int) -> tuple[float, bool, dict[str, int]]:
        """Take a step in the environment with action a.

        Args:
            a (int): Action to take.

        Returns:
            tuple[float, bool, dict[str, int]]: Reward, done flag, and info dictionary.
        """
        if a in self.action_space and a < 4:
            assert self.pos is not None and self.grid is not None
            r, c = self.pos
            self.pos = self._move(r, c, a, int(self.grid[r][c]))

        reward, done = (10, True) if (self.pos == self.end and not self.already_solved) else (0, False)
        self.already_solved = True if self.pos == self.end else self.already_solved

        # return self.render(), reward, done, self.labels()
        return float(reward), bool(done), self.labels()

    def _get_image(self) -> NDArray[uint8]:
        """Render the environment to an image."""
        assert self.grid is not None
        pos = self.pos if self.pos is not None else (-1, -1)
        if self.render_style == "mnist":
            rgb = np.concatenate(
                [
                    np.concatenate(
                        [
                            (
                                self.player_rgb * self.texture[el - 1]
                                if pos == (i, j)
                                else self.non_player_rgb * self.texture[el - 1]
                            )
                            for j, el in enumerate(row)
                        ],
                        axis=1,
                    )
                    for i, row in enumerate(self.grid)
                ],
                axis=0,
            )
        elif self.render_style == "grid_world":
            rgb = np.concatenate(
                [
                    np.concatenate(
                        [
                            (self.palette[el - 1] * self.player_rgb if pos == (i, j) else self.palette[el - 1])
                            for j, el in enumerate(row)
                        ],
                        axis=1,
                    )
                    for i, row in enumerate(self.grid)
                ],
                axis=0,
            )
        elif self.render_style == "dice":
            rgb = np.concatenate(
                [
                    np.concatenate(
                        [
                            (
                                self.player_rgb * self.texture[el - 1]
                                if pos == (i, j)
                                else self.palette[el - 1] * self.texture[el - 1]
                            )
                            for j, el in enumerate(row)
                        ],
                        axis=1,
                    )
                    for i, row in enumerate(self.grid)
                ],
                axis=0,
            )

        elif self.render_style == "beta":
            rgb = np.concatenate(
                [
                    np.concatenate(
                        [
                            (self.player_rgb + self.palette[el - 1] if pos == (i, j) else self.palette[el - 1])
                            for j, el in enumerate(row)
                        ],
                        axis=1,
                    )
                    for i, row in enumerate(self.grid)
                ],
                axis=0,
            )
        else:
            raise Exception("Unknown rendering mode.")

        rescaled = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_NEAREST)
        render = np.clip(rescaled, 0, 255)

        return render.astype(uint8)
