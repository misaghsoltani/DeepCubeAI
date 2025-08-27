# ----------------------------------------------------------------------
# Matplotlib Rubik's cube simulator
# Written by Jake Vanderplas
# Adapted from cube code written by David Hogg
#   https://github.com/davidwhogg/MagicCube

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from numpy import float32, uint8
from numpy.typing import NDArray

from deepcubeai.utils.viz_utils import Quaternion, project_points


class InteractiveCube(Axes):
    """A class to visualize a Rubik's cube in 3D using Matplotlib."""

    # Define some attributes
    base_face: NDArray[float32] = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=float32)
    stickerwidth: float = 0.9
    stickermargin: float = 0.5 * (1.0 - stickerwidth)
    stickerthickness: float = 0.001
    d1: float = 1 - stickermargin
    d2: float = 1 - 2 * stickermargin
    d3: float = 1 + stickerthickness
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

    base_face_centroid: NDArray[float32] = np.array([[0, 0, 1]], dtype=float32)
    base_sticker_centroid: NDArray[float32] = np.array([[0, 0, 1 + stickerthickness]], dtype=float32)

    def __init__(
        self,
        n: int,
        colors: NDArray[uint8],
        view: NDArray[float32] | Sequence[float] | tuple[float, float, float] = (0, 0, 10),
        fig: Figure | None = None,
        **kwargs: Any,
    ) -> None:
        self.colors: NDArray[uint8] = colors

        # Define rotation angles and axes for the six sides of the cube
        x, y, _z = np.eye(3)
        self.rots: list[Quaternion] = [Quaternion.from_v_theta(x, theta) for theta in (np.pi / 2, -np.pi / 2)]
        self.rots += [Quaternion.from_v_theta(y, theta) for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]

        rect: tuple[float, float, float, float] = (0, 0.16, 1, 0.84)
        self._move_list: list[tuple[int, int]] = []

        self.N = n
        self._prevStates: list[NDArray[float32]] = []

        self._view = view
        self._start_rot = Quaternion.from_v_theta((1, -1, 0), -np.pi / 6)

        self._grey_stickers: list[NDArray[uint8]] = []
        self._black_stickers: list[NDArray[uint8]] = []

        if fig is None:
            fig = plt.gcf()

        # disable default key press events
        callbacks = fig.canvas.callbacks.callbacks
        del callbacks["key_press_event"]

        # add some defaults, and draw axes
        kwargs.update({
            "aspect": kwargs.get("aspect", "equal"),
            "xlim": kwargs.get("xlim", (-1.7, 1.5)),
            "ylim": kwargs.get("ylim", (-1.5, 1.7)),
            "frameon": kwargs.get("frameon", False),
            "xticks": kwargs.get("xticks", []),
            "yticks": kwargs.get("yticks", []),
        })
        super().__init__(fig, rect, **kwargs)
        self.xaxis.set_major_formatter(NullFormatter())
        self.yaxis.set_major_formatter(NullFormatter())

        self._start_xlim = kwargs["xlim"]
        self._start_ylim = kwargs["ylim"]

        # Define movement for up/down arrows or up/down mouse movement
        self._ax_UD = (1, 0, 0)
        self._step_UD = 0.01

        # Define movement for left/right arrows or left/right mouse movement
        self._ax_LR = (0, -1, 0)
        self._step_LR = 0.01

        self._ax_LR_alt = (0, 0, 1)

        self._current_rot = self._start_rot  # current rotation state
        self._face_polys: list[Polygon] | None = None
        self._sticker_polys: list[Polygon] = []

        self.plastic_color = "black"

        # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
        self.face_colors: list[str] = ["w", "#ffcf00", "#ff6f00", "#cf0000", "#00008f", "#009f0f", "gray", "none"]

        self._initialize_arrays()

        self._draw_cube()
        # self._initialize_widgets()

    def set_rot(self, rot: int) -> None:
        """Sets the current rotation of the cube based on the given index."""
        if rot == 0:
            self._current_rot = Quaternion.from_v_theta((-0.53180525, 0.83020462, 0.16716299), 0.95063829)
        elif rot == 1:
            self._current_rot = Quaternion.from_v_theta((0.9248325, 0.14011997, -0.35362584), 2.49351394)

        self._draw_cube()

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

    def _project(self, pts: NDArray[float32]) -> NDArray[float32]:
        return project_points(pts, self._current_rot, self._view, [0, 1, 0])

    def _draw_cube(self) -> None:
        stickers = self._project(self._stickers)[:, :, :2]
        faces = self._project(self._faces)[:, :, :2]
        face_centroids = self._project(self._face_centroids[:, :3])
        sticker_centroids = self._project(self._sticker_centroids[:, :3])

        plastic_color = self.plastic_color
        # self._colors[np.ravel_multi_index((0,1,2),(6,N,N))] = 10
        colors = np.asarray(self.face_colors)[self.colors // (self.N**2)]
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
            for i in range(len(colors)):
                self._face_polys[i].set_xy(faces[i])
                self._face_polys[i].set_zorder(face_zorders[i])
                self._face_polys[i].set_facecolor(plastic_color)

                self._sticker_polys[i].set_xy(stickers[i])
                self._sticker_polys[i].set_zorder(sticker_zorders[i])
                self._sticker_polys[i].set_facecolor(colors[i])

        self.figure.canvas.draw()

    def new_state(self, colors: NDArray[uint8]) -> None:
        """Updates the cube's state with new colors."""
        self.colors = colors
        self._draw_cube()
