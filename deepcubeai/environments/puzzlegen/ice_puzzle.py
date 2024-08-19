import os

import cv2
import imageio
import networkx as nx
import numpy as np

from deepcubeai.environments.puzzlegen.base import PuzzleEnv


class IcePuzzle(PuzzleEnv):

    def __init__(self, ice_density=4, easy=False, render_style='human', min_sol_len=8, **kwargs):
        super(IcePuzzle, self).__init__(render_style=render_style,
                                        min_sol_len=min_sol_len,
                                        **kwargs)
        self.easy = easy
        self.ice_density = ice_density

        if self.render_style == 'human':
            self.rock_rgb = imageio.imread(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/rock.png'))
            self.ice_rgb = imageio.imread(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/ice.png'))
            self.player_rgb = imageio.imread(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/player.png'))
            # self.goal_rgb = imageio.imread(
            #   os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/goal.png'))
            self.goal_rgb = imageio.imread(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/ice.png'))
        elif self.render_style == 'grid_world':
            self.rock_rgb = np.array([[[255., 0., 0.]]])
            self.ice_rgb = np.array([[[255., 255., 255.]]])
            self.player_rgb = np.array([[[0., 255., 0.]]])
            # self.goal_rgb = np.array([[[0., 0., 255.]]])
            self.goal_rgb = np.array([[[255., 255., 255.]]])
        else:
            raise Exception('Unknown rendering mode.')

    def _can_go(self, r, c, grid):
        is_valid = (0 <= r < self.size) and (0 <= c < self.size)
        return is_valid and grid[r][c]

    def _slide(self, r, c, grid, d):
        if not self._can_go(r + self.y[d], c + self.x[d], grid):
            return r, c
        return self._slide(r + self.y[d], c + self.x[d], grid, d)

    def _create_level(self):

        for _ in range(self.max_tries):
            start = (0, self.rng.randint(0, self.size - 1))
            end = (self.size - 1, self.rng.randint(0, self.size - 1))
            grid = [[self.rng.randint(0, self.ice_density) for _ in range(self.size)]
                    for _ in range(self.size)]
            grid[0][start[1]] = 1
            grid[self.size - 1][end[1]] = 1
            q = [(start, [])]
            n = start
            z = {}
            while q and n != end:
                n, path = q.pop()
                for d in range(4):
                    N = self._slide(*n, grid, d)
                    if self._can_go(n[0] + self.y[d], n[1] + self.x[d], grid) and N not in z:
                        q[:0] = [(N, path + [d])]
                        z[N] = 1

            if (n == end) * len(path) > self.min_sol_len:
                if self.easy:
                    break
                else:
                    # the shortest solution is longer than required, now we check that you can get
                    # stuck a strongly connected component is reachable from the start, but not
                    # connected to the end
                    g = nx.MultiDiGraph()
                    count = 0
                    node_id = {}
                    for i in range(self.size):
                        for j in range(self.size):
                            if grid[i][j]:
                                node_id[(i, j)] = count
                                count += 1
                    edges = []
                    for pos in node_id.keys():
                        for d in range(4):
                            edges.append((node_id[pos], node_id[self._slide(*pos, grid, d)]))
                    g.add_nodes_from(np.arange(count))
                    g.add_edges_from(edges)
                    c = nx.condensation(g)

                    def hard_enough(c, start, target):
                        for n in c.nodes:
                            if nx.has_path(c, start, n) and not nx.has_path(c, n, target):
                                return True
                        return False

                    if hard_enough(c, c.graph['mapping'][node_id[start]],
                                   c.graph['mapping'][node_id[end]]):
                        break

        self.grid = (np.array(grid) != 0).astype(np.uint8)
        self.pos = start
        self.end = end
        self.solution = path

    def _reset(self):
        self._create_level()
        return self.render()

    def _step(self, a):
        if a in self.action_space and a < 4:
            self.pos = self._slide(*self.pos, self.grid, a)

        reward, done = (10.,
                        True) if (self.pos == self.end and not self.already_solved) else (0, False)
        self.already_solved = True if self.pos == self.end else self.already_solved

        # return self.render(), reward, done, self.labels()
        return reward, done, self.labels()

    def _get_image(self):
        rgb = np.concatenate([
            np.concatenate([
                self.rock_rgb if not el else
                (self.player_rgb if self.pos == (i, j) else
                 (self.goal_rgb if self.end == (i, j) else self.ice_rgb))
                for j, el in enumerate(row)
            ],
                           axis=1) for i, row in enumerate(self.grid)
        ],
                             axis=0)
        rescaled = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_NEAREST)
        render = np.clip(rescaled, 0, 255)
        return render.astype(np.uint8)
