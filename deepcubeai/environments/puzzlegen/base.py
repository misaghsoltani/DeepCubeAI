import random

import numpy as np
from gym.spaces import Box, Discrete


class PuzzleEnv:

    def __init__(self,
                 seed=None,
                 size=8,
                 render_style='grid_world',
                 min_sol_len=1,
                 max_tries=1000000):

        self.x = (0, 1, -1, 0)
        self.y = (-1, 0, 0, 1)
        self.action_space = Discrete(5)
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 3), dtype="uint8")

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

    def reset(self):
        self.needs_reset = False
        self.rng = random.Random(self.seed)
        self.already_solved = False
        return self._reset()

    def _reset(self):
        raise NotImplementedError()

    def step(self, a):
        if self.needs_reset:
            raise Exception('Environment needs to be reset.')
        if self.is_closed:
            raise Exception('Environment is closed.')
        assert isinstance(a, int)
        return self._step(a)

    def _step(self, a):
        raise NotImplementedError()

    def render(self, mode='human'):
        if self.needs_reset:
            raise Exception('Environment needs to be reset.')
        if self.is_closed:
            raise Exception('Environment is closed.')
        return self._get_image()

    def _get_image(self):
        raise NotImplementedError()

    def close(self):
        if self.is_closed:
            return
        self.is_closed = True
        self.grid = None
        self.pos = None
        self.start = None
        self.end = None
        self.solution = None
        self.already_solved = None

    def seed(self, seed=None):
        self.seed = seed

    def get_solution(self):
        if self.needs_reset:
            raise Exception('Environment needs to be reset.')
        if self.is_closed:
            raise Exception('Environment is closed.')
        return self.solution

    def labels(self):
        if self.needs_reset:
            raise Exception('Environment needs to be reset.')
        if self.is_closed:
            raise Exception('Environment is closed.')
        return {"player_x": self.pos[0], "player_y": self.pos[1]}
