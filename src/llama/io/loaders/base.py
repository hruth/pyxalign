from typing import Optional
import numpy as np


class StandardData:
    """Standard format that is required for doing laminography alignment."""

    def __init__(
        self,
        projections: dict[int, np.ndarray],
        angles: np.ndarray,
        scan_numbers: Optional[np.ndarray],
        file_paths: Optional[list],
    ):
        self.projections = projections
        self.angles = angles
        self.scan_numbers = scan_numbers
        self.file_paths = file_paths
