import numpy as np
from pyxalign.api.types import r_type


def symmetric_gaussian_2d(shape: tuple, amplitude: float, sigma: float):
    """
    Generate a symmetric 2D Gaussian.

    Parameters:
    - shape: tuple of ints (height, width), size of the output array
    - amplitude: float, peak amplitude of the Gaussian
    - sigma: float, standard deviation (same in x and y)

    Returns:
    - 2D numpy array of shape `shape` containing the Gaussian
    """
    height, width = shape
    y = np.linspace(0, height - 1, height, dtype=r_type)
    x = np.linspace(0, width - 1, width, dtype=r_type)
    x, y = np.meshgrid(x, y)

    x0 = (width - 1) / 2
    y0 = (height - 1) / 2

    gaussian = amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    return gaussian.astype(r_type)
