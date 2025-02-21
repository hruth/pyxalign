from typing import Optional
import numpy as np


class StandardData:
    """Standard format that is required for doing laminography alignment."""

    def __init__(
        self,
        projections: dict[int, np.ndarray],
        angles: np.ndarray,
        scan_numbers: np.ndarray,
        file_paths: Optional[list] = None,
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
