"""Array rotation and baseline geometry."""

import numpy as np


def rotation_angles(n_rotations: int, n_samples: int) -> np.ndarray:
    """Array rotation angle at the start of each detector integration.

    :param n_rotations: number of full array rotations in the observation
    :param n_samples: number of detector integrations
    :return: angles in rad, shape (n_samples,), from 0 up to but excluding
        2 pi n_rotations
    """
    return 2 * np.pi * n_rotations * np.arange(n_samples) / n_samples


def rotate_positions(positions: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotate collector positions counter-clockwise about the array centre.

    :param positions: collector positions in m, shape (n_collectors, 2)
    :param angles: rotation angles in rad, shape (n_t,)
    :return: rotated positions in m, shape (n_t, n_collectors, 2)
    """
    c = np.cos(angles)[:, np.newaxis]
    s = np.sin(angles)[:, np.newaxis]
    x = c * positions[:, 0] - s * positions[:, 1]
    y = s * positions[:, 0] + c * positions[:, 1]
    return np.stack([x, y], axis=-1)


def baselines(positions: np.ndarray) -> np.ndarray:
    """Baseline vectors x_jk = x_j - x_k for all collector pairs.

    :param positions: collector positions in m, shape (n_t, n_collectors, 2)
    :return: baselines in m, shape (n_t, n_collectors, n_collectors, 2);
        element [t, j, k] is x_j - x_k
    """
    return positions[:, :, np.newaxis, :] - positions[:, np.newaxis, :, :]
