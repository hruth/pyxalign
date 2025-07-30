from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


class StandardData:
    """Standard format that is required for doing laminography alignment."""

    def __init__(
        self,
        projections: dict[int, np.ndarray],
        angles: np.ndarray,
        scan_numbers: np.ndarray,
        file_paths: Optional[dict] = None,
        probe_positions: Optional[dict[int, np.ndarray]] = None,
        probe: Optional[np.ndarray] = None,
        pixel_size: Optional[float] = None,
    ):
        self.projections = projections
        self.angles = angles
        self.scan_numbers = scan_numbers
        self.file_paths = file_paths
        self.probe_positions = probe_positions
        self.probe = probe
        self.pixel_size = pixel_size

    def drop_scans(self, scan_numbers_to_drop: list[int]):
        # Update dictionaries
        for scan_number in scan_numbers_to_drop:
            del self.projections[scan_number]
            if self.probe_positions is not None:
                del self.probe_positions[scan_number]
            if self.file_paths is not None:
                del self.file_paths[scan_number]
        keep_idx = [
            i for i, scan in enumerate(self.scan_numbers) if scan not in scan_numbers_to_drop
        ]
        # Update arrays
        self.scan_numbers = self.scan_numbers[keep_idx]
        self.angles = self.angles[keep_idx]

    def plot_sample_projection(self, index: int = 0):
        scan_number = list(self.scan_numbers)[index]
        plt.title(f"Scan {scan_number}")
        plt.imshow(np.angle(self.projections[scan_number]), cmap="bone")
        plt.show()

    def get_minimum_size_for_projection_array(self) -> np.ndarray:
        return np.array(
            (
                np.max([v.shape[0] for v in self.projections.values()]),
                np.max([v.shape[1] for v in self.projections.values()]),
            )
        ).astype(int)
