from typing import List, Tuple, Union

import numpy as np


class Quaternion:
    """Quaternion Rotation:

    Class to aid in representing 3D rotations via quaternions.
    """

    @classmethod
    def from_v_theta(cls, v: Union[np.ndarray, List[float]],
                     theta: Union[np.ndarray, List[float]]) -> "Quaternion":
        """
        Construct quaternions from unit vectors v and rotation angles theta.

        Args:
            v (Union[np.ndarray, List[float]]): Array of vectors, last dimension 3. Vectors will
                be normalized.
            theta (Union[np.ndarray, List[float]]): Array of rotation angles in radians,
                shape = v.shape[:-1].

        Returns:
            Quaternion: Quaternion representing the rotations.
        """
        theta = np.asarray(theta)
        v = np.asarray(v)
        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)

        v = v * s / np.sqrt(np.sum(v * v, -1))
        x_shape = v.shape[:-1] + (4, )

        x = np.ones(x_shape).reshape(-1, 4)
        x[:, 0] = c.ravel()
        x[:, 1:] = v.reshape(-1, 3)
        x = x.reshape(x_shape)

        return cls(x)

    def __init__(self, x: Union[np.ndarray, List[float]]):
        """
        Initializes the Quaternion.

        Args:
            x (Union[np.ndarray, List[float]]): The quaternion components.
        """
        self.x = np.asarray(x, dtype=float)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Quaternion.

        Returns:
            str: String representation of the Quaternion.
        """
        return "Quaternion:\n" + self.x.__repr__()

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """
        Multiplies two quaternions.

        Args:
            other (Quaternion): The other quaternion to multiply with.

        Returns:
            Quaternion: The product of the two quaternions.
        """
        sxr = self.x.reshape(self.x.shape[:-1] + (4, 1))
        oxr = other.x.reshape(other.x.shape[:-1] + (1, 4))

        prod = sxr * oxr
        return_shape = prod.shape[:-1]
        prod = prod.reshape((-1, 4, 4)).transpose((1, 2, 0))

        ret = np.array(
            [
                (prod[0, 0] - prod[1, 1] - prod[2, 2] - prod[3, 3]),
                (prod[0, 1] + prod[1, 0] + prod[2, 3] - prod[3, 2]),
                (prod[0, 2] - prod[1, 3] + prod[2, 0] + prod[3, 1]),
                (prod[0, 3] + prod[1, 2] - prod[2, 1] + prod[3, 0]),
            ],
            dtype=float,
            order="F",
        ).T
        return self.__class__(ret.reshape(return_shape))

    def as_v_theta(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the v, theta equivalent of the (normalized) quaternion.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The unit vector and rotation angle.
        """
        x = self.x.reshape((-1, 4)).T

        # compute theta
        norm = np.sqrt((x**2).sum(0))
        theta = 2 * np.arccos(x[0] / norm)

        # compute the unit vector
        v = np.array(x[1:], order="F", copy=True)
        v /= np.sqrt(np.sum(v**2, 0))

        # reshape the results
        v = v.T.reshape(self.x.shape[:-1] + (3, ))
        theta = theta.reshape(self.x.shape[:-1])

        return v, theta

    def as_rotation_matrix(self) -> np.ndarray:
        """
        Returns the rotation matrix of the (normalized) quaternion.

        Returns:
            np.ndarray: The rotation matrix.
        """
        v, theta = self.as_v_theta()

        shape = theta.shape
        theta = theta.reshape(-1)
        v = v.reshape(-1, 3).T
        c = np.cos(theta)
        s = np.sin(theta)

        mat = np.array(
            [
                [
                    v[0] * v[0] * (1.0 - c) + c,
                    v[0] * v[1] * (1.0 - c) - v[2] * s,
                    v[0] * v[2] * (1.0 - c) + v[1] * s,
                ],
                [
                    v[1] * v[0] * (1.0 - c) + v[2] * s,
                    v[1] * v[1] * (1.0 - c) + c,
                    v[1] * v[2] * (1.0 - c) - v[0] * s,
                ],
                [
                    v[2] * v[0] * (1.0 - c) - v[1] * s,
                    v[2] * v[1] * (1.0 - c) + v[0] * s,
                    v[2] * v[2] * (1.0 - c) + c,
                ],
            ],
            order="F",
        ).T
        return mat.reshape(shape + (3, 3))

    def rotate(self, points: np.ndarray) -> np.ndarray:
        """
        Rotates the given points using the quaternion.

        Args:
            points (np.ndarray): The points to rotate.

        Returns:
            np.ndarray: The rotated points.
        """
        rot_mat = self.as_rotation_matrix()
        return np.dot(points, rot_mat.T)


def project_points(points: np.ndarray, q: Quaternion, view: np.ndarray,
                   vertical: np.ndarray) -> np.ndarray:
    """Project points using a quaternion q and a view v.

    Args:
        points (np.ndarray): Array of last-dimension 3.
        q (Quaternion): Quaternion representation of the rotation.
        view (np.ndarray): Length-3 vector giving the point of view.
        vertical (np.ndarray): Direction of y-axis for view. An error will be raised if it is
            parallel to the view.

    Returns:
        np.ndarray: Array of projected points: same shape as points.
    """
    if vertical is None:
        vertical = [0, 1, 0]
    points = np.asarray(points)
    view = np.asarray(view)

    xdir = np.cross(vertical, view).astype(float)

    if np.all(xdir == 0):
        raise ValueError("vertical is parallel to v")

    xdir /= np.sqrt(np.dot(xdir, xdir))

    # get the unit vector corresponding to vertical
    ydir = np.cross(view, xdir)
    ydir /= np.sqrt(np.dot(ydir, ydir))

    # normalize the viewer location: this is the z-axis
    v2 = np.dot(view, view)
    zdir = view / np.sqrt(v2)

    # rotate the points
    rot_mat = q.as_rotation_matrix()
    r_pts = np.dot(points, rot_mat.T)

    # project the points onto the view
    dpoint = r_pts - view
    dpoint_view = np.dot(dpoint, view).reshape(dpoint.shape[:-1] + (1, ))
    dproj = -dpoint * v2 / dpoint_view

    trans = list(range(1, dproj.ndim)) + [0]
    return np.array([np.dot(dproj, xdir),
                     np.dot(dproj, ydir), -np.dot(dpoint, zdir)]).transpose(trans)
