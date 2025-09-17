from typing import Optional
import numpy as np

import matplotlib.pyplot as plt
from pyxalign.transformations.functions import rotate_positions, shear_positions


class ProbePositions:
    def __init__(self, positions: list[np.ndarray], center_pixel: np.ndarray):
        self.data = [position + center_pixel for position in positions]

    def rotate_positions(
        self, angle: float, center_pixel: np.ndarray, new_center_pixel: Optional[np.ndarray] = None
    ):
        for i, positions in enumerate(self.data):
            self.data[i] = rotate_positions(positions, angle, center_pixel, new_center_pixel)

    def shear_positions(self, angle: float, center_pixel: np.ndarray):
        for i, positions in enumerate(self.data):
            self.data[i] = shear_positions(positions, angle, center_pixel, axis=1)

    def rescale_positions(self, scale: int):
        for i, positions in enumerate(self.data):
            self.data[i] = positions / scale

    def shift_positions(self, shift: np.ndarray):
        for i in range(len(self.data)):
            self.data[i][:, 0] += shift[i, 1]
            self.data[i][:, 1] += shift[i, 0]

    def crop_positions(self, x_max: int, y_max: int):
        for i, positions in enumerate(self.data):
            in_bounds_idx_y = (positions[:, 0] > 0) * (positions[:, 0] < y_max)
            in_bounds_idx_x = (positions[:, 1] > 0) * (positions[:, 1] < x_max)
            in_bounds_idx = in_bounds_idx_x * in_bounds_idx_y
            self.data[i] = self.data[i][in_bounds_idx]

    def plot_positions(self, index: int, color: str = "m", linestyle: str = "-"):
        positions = self.data[index]
        plt.plot(positions[:, 1], positions[:, 0], color=color, linestyle=linestyle)
