import numpy as np

import matplotlib.pyplot as plt
from llama.transformations.functions import rotate_positions, shear_positions


class ProbePositions:
    def __init__(self, positions: list[np.ndarray], center_pixel: np.ndarray):
        self.data = [position + center_pixel for position in positions]

    def rotate_positions(self, angle: float, center_pixel: np.ndarray):
        for i, positions in enumerate(self.data):
            self.data[i] = rotate_positions(positions, angle, center_pixel)

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

    def plot_positions(self, index: int, color: str = "m", linestyle: str = "-"):
        positions = self.data[index]
        plt.plot(positions[:, 1], positions[:, 0], color=color, linestyle=linestyle)
