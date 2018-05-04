from math import pi

import numpy as np


def get_ellipse(axes: np.ndarray, resolution: int=21) -> np.ndarray:
    """get polygonal ellipses for params in numpy arrays
    Args:
        axes: np.array([[a, b], ...])
            a - horizontal axis of the ellipse, in visual degrees
            b - vertical axis of the ellipse, in visual degrees
        resolution: number of polygons in the ellipse
    Returns:
        a 2d array, each row is an array of complex that forms the circumvent
    """
    angles = np.exp(-1j * np.linspace(-pi, pi, resolution + 1, endpoint=True))
    ray = np.outer((axes[:, 0] + axes[:, 1]) * 0.5, angles) + \
        np.outer((axes[:, 0] - axes[:, 1]) * 0.5, np.flipud(angles))
    return ray
