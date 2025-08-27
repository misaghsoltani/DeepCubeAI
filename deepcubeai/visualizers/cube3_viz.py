# ----------------------------------------------------------------------
# Matplotlib Rubik's cube simulator
# Written by Jake Vanderplas
# Adapted from cube code written by David Hogg
#   https://github.com/davidwhogg/MagicCube

from __future__ import annotations

import argparse
from collections.abc import Callable
import re
from typing import Any, cast

from matplotlib import widgets
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from numpy import float32
from numpy.typing import NDArray

from deepcubeai.environments.cube3 import Cube3, Cube3State
from deepcubeai.utils.viz_utils import Quaternion, project_points


class InteractiveCube(Axes):
    """A class to visualize a Rubik's cube in 3D using Matplotlib."""

    # Define some attributes
    base_face: NDArray[float32] = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=float32)
    stickerwidth: float = 0.9
    stickermargin: float = 0.5 * (1.0 - stickerwidth)
    stickerthickness: float = 0.001
    d1: float
    d2: float
    d3: float
    (d1, d2, d3) = (1 - stickermargin, 1 - 2 * stickermargin, 1 + stickerthickness)
    base_sticker: NDArray[float32] = np.array(
        [
            [d1, d2, d3],
            [d2, d1, d3],
            [-d2, d1, d3],
            [-d1, d2, d3],
            [-d1, -d2, d3],
            [-d2, -d1, d3],
            [d2, -d1, d3],
            [d1, -d2, d3],
            [d1, d2, d3],
        ],
        dtype=float32,
    )

    base_face_centroid: NDArray[float32] = np.array([[0, 0, 1]])
    base_sticker_centroid: NDArray[float32] = np.array([[0, 0, 1 + stickerthickness]])

    def __init__(
        self,
        n: int,
        state: Cube3State,
        view: tuple[float, float, float] = (0.0, 0.0, 10.0),
        fig: Figure | None = None,
        **kwargs: Any,
    ) -> None:
        self.state: Cube3State = state

        # Define rotation angles and axes for the six sides of the cube
        x, y, _z = np.eye(3)
        self.rots: list[Quaternion] = [Quaternion.from_v_theta(x, theta) for theta in (np.pi / 2, -np.pi / 2)]
        self.rots += [Quaternion.from_v_theta(y, theta) for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]

        rect: tuple[float, float, float, float] = (0.0, 0.16, 1.0, 0.84)
        self._move_list: list[tuple[str, int, int]] = []

        self.N = n
        self._prevStates: list[Cube3State] = []

        self.env: Cube3 = Cube3()

        self._view = view
        self._start_rot = Quaternion.from_v_theta((1.0, -1.0, 0.0), -np.pi / 6)

        self._grey_stickers: list[int] = []
        self._black_stickers: list[int] = []

        if fig is None:
            fig = plt.gcf()

        # disable default key press events
        callbacks = fig.canvas.callbacks.callbacks
        # Some backends may not have the key_press_event; guard access
        if "key_press_event" in callbacks:
            del callbacks["key_press_event"]

        # add some defaults, and draw axes
        options_aspect: Any = kwargs.get("aspect", "equal")
        options_xlim: tuple[float, float] = kwargs.get("xlim", (-1.7, 1.5))
        options_ylim: tuple[float, float] = kwargs.get("ylim", (-1.5, 1.7))
        options_frameon: bool = kwargs.get("frameon", False)
        options_xticks: list[Any] = kwargs.get("xticks", [])
        options_yticks: list[Any] = kwargs.get("yticks", [])

        super().__init__(fig, rect)
        self.set_aspect(options_aspect)
        self.set_xlim(*options_xlim)
        self.set_ylim(*options_ylim)
        self.set_frame_on(options_frameon)
        self.set_xticks(options_xticks)
        self.set_yticks(options_yticks)
        self.xaxis.set_major_formatter(NullFormatter())
        self.yaxis.set_major_formatter(NullFormatter())

        self._start_xlim = options_xlim
        self._start_ylim = options_ylim

        # Define movement for up/down arrows or up/down mouse movement
        self._ax_UD = (1, 0, 0)
        self._step_UD = 0.01

        # Define movement for left/right arrows or left/right mouse movement
        self._ax_LR = (0, -1, 0)
        self._step_LR = 0.01

        self._ax_LR_alt = (0, 0, 1)

        # Internal state variable
        self._active = False  # true when mouse is over axes
        self._button1 = False  # true when button 1 is pressed
        self._button2 = False  # true when button 2 is pressed
        self._event_xy: tuple[float, float] | None = None  # store xy position of mouse event
        self._tab = False  # tab key pressed
        self._digits = []  # type: list[str]

        self._current_rot = self._start_rot  # current rotation state
        self._face_polys: list[Polygon] | None = None
        self._sticker_polys: list[Polygon] | None = None

        self.plastic_color = "black"

        # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
        self.face_colors: list[str] = ["w", "#ffcf00", "#ff6f00", "#cf0000", "#00008f", "#009f0f", "gray", "none"]

        self._initialize_arrays()

        # connect some GUI events
        self.figure.canvas.mpl_connect("button_press_event", cast(Callable[..., Any], self._mouse_press))
        self.figure.canvas.mpl_connect("button_release_event", cast(Callable[..., Any], self._mouse_release))
        self.figure.canvas.mpl_connect("motion_notify_event", cast(Callable[..., Any], self._mouse_motion))
        self.figure.canvas.mpl_connect("key_press_event", cast(Callable[..., Any], self._key_press))
        self.figure.canvas.mpl_connect("key_release_event", cast(Callable[..., Any], self._key_release))

        self._draw_cube()
        # self._initialize_widgets()

    def set_rot(self, rot: int) -> None:
        """Set one of a few predefined viewing rotations by index and redraw."""
        if rot == 0:
            self._current_rot = Quaternion.from_v_theta((-0.53180525, 0.83020462, 0.16716299), 0.95063829)
        elif rot == 1:
            self._current_rot = Quaternion.from_v_theta((0.9248325, 0.14011997, -0.35362584), 2.49351394)

        self._draw_cube()

    def rotate(self, rot: Quaternion) -> None:
        """Apply an incremental rotation to the current view quaternion."""
        self._current_rot *= rot

        # print(self._start_rot.as_v_theta())
        # print(self._current_rot.as_v_theta())

    def _initialize_arrays(self) -> None:
        # initialize centroids, faces, and stickers.  We start with a
        # base for each one, and then translate & rotate them into position.

        # Define N^2 translations for each face of the cube
        cubie_width = 2.0 / self.N
        translations = np.array([
            [[-1 + (i + 0.5) * cubie_width, -1 + (j + 0.5) * cubie_width, 0]]
            for i in range(self.N)
            for j in range(self.N)
        ])

        # Create arrays for centroids, faces, stickers
        face_centroids = []
        faces = []
        sticker_centroids = []
        stickers = []
        colors = []

        factor = np.array([1.0 / self.N, 1.0 / self.N, 1])

        for i in range(6):
            rot_mat = self.rots[i].as_rotation_matrix()
            faces_t = np.dot(factor * self.base_face + translations, rot_mat.T)
            stickers_t = np.dot(factor * self.base_sticker + translations, rot_mat.T)
            face_centroids_t = np.dot(self.base_face_centroid + translations, rot_mat.T)
            sticker_centroids_t = np.dot(self.base_sticker_centroid + translations, rot_mat.T)
            # colors_i = i + np.zeros(face_centroids_t.shape[0], dtype=int)
            colors_i = np.arange(i * face_centroids_t.shape[0], (i + 1) * face_centroids_t.shape[0])

            # append face ID to the face centroids for lex-sorting
            face_centroids_t = np.hstack([face_centroids_t.reshape(-1, 3), colors_i[:, None]])
            sticker_centroids_t = sticker_centroids_t.reshape((-1, 3))

            faces.append(faces_t)
            face_centroids.append(face_centroids_t)
            stickers.append(stickers_t)
            sticker_centroids.append(sticker_centroids_t)

            colors.append(colors_i)

        self._face_centroids = np.vstack(face_centroids)
        self._faces = np.vstack(faces)
        self._sticker_centroids = np.vstack(sticker_centroids)
        self._stickers = np.vstack(stickers)

    def reset(self) -> None:
        """Reset the cube state to the canonical goal state and redraw."""
        self.state = self.env._generate_canonical_goal_states(1)[0]

    def _initialize_widgets(self) -> None:
        self._ax_reset = self.figure.add_axes((0.75, 0.05, 0.2, 0.075))
        self._btn_reset = widgets.Button(self._ax_reset, "Reset View")
        self._btn_reset.on_clicked(lambda _event: self._reset_view())

        self._ax_solve = self.figure.add_axes((0.55, 0.05, 0.2, 0.075))
        self._btn_solve = widgets.Button(self._ax_solve, "Solve Cube")
        self._btn_solve.on_clicked(lambda _event: self._solve_cube())

    def _project(self, pts: np.ndarray) -> np.ndarray:
        return project_points(pts, self._current_rot, self._view, [0.0, 1.0, 0.0])

    def _draw_cube(self) -> None:
        stickers = self._project(self._stickers)[:, :, :2]
        faces = self._project(self._faces)[:, :, :2]
        face_centroids = self._project(self._face_centroids[:, :3])
        sticker_centroids = self._project(self._sticker_centroids[:, :3])

        plastic_color = self.plastic_color
        # self._colors[np.ravel_multi_index((0,1,2),(6,N,N))] = 10
        colors = np.asarray(self.face_colors)[self.state.colors // (self.N**2)]
        for idx in self._grey_stickers:
            colors[idx] = "grey"
        for idx in self._black_stickers:
            colors[idx] = "k"

        face_zorders = -face_centroids[:, 2]
        sticker_zorders = -sticker_centroids[:, 2]

        if self._face_polys is None:
            # initial call: create polygon objects and add to axes
            self._face_polys = []
            self._sticker_polys = []

            for i in range(len(colors)):
                fp = Polygon(faces[i], facecolor=plastic_color, zorder=face_zorders[i])
                sp = Polygon(stickers[i], facecolor=colors[i], zorder=sticker_zorders[i])

                self._face_polys.append(fp)
                self._sticker_polys.append(sp)
                self.add_patch(fp)
                self.add_patch(sp)
        else:
            # subsequent call: update the polygon objects
            assert self._face_polys is not None and self._sticker_polys is not None
            for i in range(len(colors)):
                self._face_polys[i].set_xy(faces[i])
                self._face_polys[i].set_zorder(face_zorders[i])
                self._face_polys[i].set_facecolor(plastic_color)

                self._sticker_polys[i].set_xy(stickers[i])
                self._sticker_polys[i].set_zorder(sticker_zorders[i])
                self._sticker_polys[i].set_facecolor(colors[i])

        self.figure.canvas.draw()

    def new_state(self, state: Cube3State) -> None:
        """Replace the current state with a new one and redraw."""
        self.state = state
        self._move_list = []
        self._draw_cube()

    def rand_action(self) -> int:
        """Sample a random valid action from the environment for this state."""
        return self.env.rand_action([self.state])[0]

    def next_state(self, action: int) -> None:
        """Advance the cube by applying an action and redraw."""
        self.state = cast(Cube3State, self.env.next_state([self.state], [action])[0][0])
        self._draw_cube()

    def rotate_face(self, f: str, n: int = 1, layer: int = 0) -> None:
        """Rotate a face by n quarter-turns on the specified layer and redraw."""
        self._move_list.append((f, n, layer))

        if not np.allclose(n, 0):
            move: str = f"{f}{n}"
            move_idx: int = np.where(np.array(self.env.moves) == np.array(move))[0][0]
            self.state = cast(Cube3State, self.env.next_state([self.state], [move_idx])[0][0])
            self._draw_cube()

    def move(self, move: list[int], draw: bool = True) -> None:
        """Apply a move sequence (list of action indices); optionally redraw."""
        self.state = cast(Cube3State, self.env.next_state([self.state], move)[0][0])
        if draw:
            self._draw_cube()

    def _reset_view(self) -> None:
        self.set_xlim(self._start_xlim)
        self.set_ylim(self._start_ylim)
        self._current_rot = self._start_rot
        self._draw_cube()

    def _solve_cube(self) -> None:
        move_list = self._move_list[:]
        for face, n, layer in move_list[::-1]:
            self.rotate_face(face, -n, layer)
        self._move_list = []

    def _key_press(self, event: Any) -> None:
        if event.key == "tab":
            self._tab = True
        elif event.key.isdigit():
            self._digits.append(event.key)
        elif event.key == "right":
            ax_lr = self._ax_LR_alt if self._tab else self._ax_LR
            self.rotate(Quaternion.from_v_theta(ax_lr, 5 * self._step_LR))
        elif event.key == "left":
            ax_lr = self._ax_LR_alt if self._tab else self._ax_LR
            self.rotate(Quaternion.from_v_theta(ax_lr, -5 * self._step_LR))
        elif event.key == "up":
            self.rotate(Quaternion.from_v_theta(self._ax_UD, 5 * self._step_UD))
        elif event.key == "down":
            self.rotate(Quaternion.from_v_theta(self._ax_UD, -5 * self._step_UD))
        elif event.key.upper() in "LRUDBF":
            direction = -1 if self._tab else 1

            self.rotate_face(event.key.upper(), direction)

        elif event.key.upper() == "P":
            cast(Figure, self.figure).savefig("snapshot.jpg", transparent=True)

        elif event.key.upper() == "S":
            self._solve_cube()

        elif event.key.upper() == "K":
            idx = int("".join(self._digits))
            self._black_stickers = [idx]

            self._digits = []

        self._draw_cube()

        goal_state = self.env._generate_canonical_goal_states(1)[0]
        is_solved: bool = bool(self.env.is_solved([self.state], [goal_state])[0])
        if is_solved:
            print("SOLVED!")
            self._move_list = []
            self._prevStates = []

    def _key_release(self, event: Any) -> None:
        if event.key == "tab":
            self._tab = False

    def _mouse_press(self, event: Any, event_x: float | None = None, event_y: float | None = None) -> None:
        if event_x is not None and event_y is not None:
            self._event_xy = (event_x, event_y)
            self._button1 = True
        else:
            self._event_xy = (event.x, event.y)
            if event.button == 1:
                self._button1 = True
            elif event.button == 3:
                self._button2 = True

    def _mouse_release(self, event: Any) -> None:
        self._event_xy = None
        if event.button == 1:
            self._button1 = False
        elif event.button == 3:
            self._button2 = False

    def _mouse_motion(self, event: Any, event_x: float | None = None, event_y: float | None = None) -> None:
        if self._button1 or self._button2:
            # Ensure we have a previous event position before computing deltas
            if self._event_xy is None:
                if event_x is not None and event_y is not None:
                    self._event_xy = (event_x, event_y)
                else:
                    self._event_xy = (event.x, event.y)
                return

            if event_x is not None and event_y is not None:
                dx = event_x - self._event_xy[0]
                dy = event_y - self._event_xy[1]
                self._event_xy = (event_x, event_y)
            else:
                dx = event.x - self._event_xy[0]
                dy = event.y - self._event_xy[1]
                self._event_xy = (event.x, event.y)

            if self._button1:
                ax_lr = self._ax_LR_alt if self._tab else self._ax_LR
                rot1 = Quaternion.from_v_theta(self._ax_UD, self._step_UD * dy)
                rot2 = Quaternion.from_v_theta(ax_lr, self._step_LR * dx)
                self.rotate(rot1 * rot2)

                self._draw_cube()

            if self._button2:
                factor = 1 - 0.003 * (dx + dy)
                xlim = self.get_xlim()
                ylim = self.get_ylim()
                self.set_xlim(factor * xlim[0], factor * xlim[1])
                self.set_ylim(factor * ylim[0], factor * ylim[1])

                self.figure.canvas.draw()


def main() -> None:
    """Run an interactive Rubik's Cube viewer for Cube3 in a Matplotlib window."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, default=None, help="")
    parser.add_argument("--moves", type=str, default=None, help="")
    args = parser.parse_args()

    env: Cube3 = Cube3()
    state: Cube3State = env._generate_canonical_goal_states(1)[0]
    if args.init is not None:
        state_rep = re.sub("[^0-9,]", "", args.init)
        state = Cube3State(np.array([int(x) for x in state_rep.split(",")]))

    cube_len = 3

    fig = plt.figure(figsize=(5, 5))
    interactive_cube = InteractiveCube(cube_len, state)
    fig.add_axes(interactive_cube)

    plt.show()


if __name__ == "__main__":
    main()
