from __future__ import annotations

from abc import ABC, abstractmethod
import random

from gymnasium.spaces import Box, Discrete
from numpy import uint8
from numpy.typing import NDArray


class PuzzleEnv(ABC):
    """Base class for puzzle environments."""

    seed: int | None
    rng: random.Random
    size: int
    render_style: str
    min_sol_len: int
    max_tries: int
    needs_reset: bool
    is_closed: bool
    grid: NDArray[uint8] | None
    pos: tuple[int, int] | None
    start: tuple[int, int] | None
    end: tuple[int, int] | None
    solution: list[int] | None
    already_solved: bool | None
    x: tuple[int, int, int, int]
    y: tuple[int, int, int, int]
    action_space: Discrete
    observation_space: Box

    def __init__(
        self,
        seed: int | None = None,
        size: int = 8,
        render_style: str = "grid_world",
        min_sol_len: int = 1,
        max_tries: int = 1000000,
    ) -> None:
        self.x = (0, 1, -1, 0)
        self.y = (-1, 0, 0, 1)
        self.action_space = Discrete(5)
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 3), dtype=uint8)

        self.seed = seed
        self.rng = random.Random(seed)
        self.size = size
        self.render_style = render_style
        self.min_sol_len = min_sol_len
        self.max_tries = max_tries

        self.needs_reset = True
        self.is_closed = False
        self.grid = None
        self.pos = None
        self.start = None
        self.end = None
        self.solution = None
        self.already_solved = None

    def reset(self) -> NDArray[uint8]:
        """Reset the environment to initial state."""
        self.needs_reset = False
        self.rng = random.Random(self.seed)
        self.already_solved = False
        return self._reset()

    @abstractmethod
    def _reset(self) -> NDArray[uint8]: ...

    def step(self, a: int) -> tuple[float, bool, dict[str, int]]:
        """Take a step in the environment with action a."""
        if self.needs_reset:
            raise Exception("Environment needs to be reset.")
        if self.is_closed:
            raise Exception("Environment is closed.")
        assert isinstance(a, int)
        return self._step(a)

    @abstractmethod
    def _step(self, a: int) -> tuple[float, bool, dict[str, int]]: ...

    def render(self, mode: str = "human") -> NDArray[uint8]:
        """Render the environment."""
        if self.needs_reset:
            raise Exception("Environment needs to be reset.")
        if self.is_closed:
            raise Exception("Environment is closed.")
        return self._get_image()

    @abstractmethod
    def _get_image(self) -> NDArray[uint8]: ...

    def close(self) -> None:
        """Close the environment."""
        if self.is_closed:
            return
        self.is_closed = True
        self.grid = None
        self.pos = None
        self.start = None
        self.end = None
        self.solution = None
        self.already_solved = None

    def get_solution(self) -> list[int] | None:
        """Get the solution path for the current puzzle state."""
        if self.needs_reset:
            raise Exception("Environment needs to be reset.")
        if self.is_closed:
            raise Exception("Environment is closed.")
        return self.solution

    def labels(self) -> dict[str, int]:
        """Get labels for the current state."""
        if self.needs_reset:
            raise Exception("Environment needs to be reset.")
        if self.is_closed:
            raise Exception("Environment is closed.")
        px, py = (0, 0)
        if self.pos is not None:
            px, py = self.pos
        return {"player_x": px, "player_y": py}
