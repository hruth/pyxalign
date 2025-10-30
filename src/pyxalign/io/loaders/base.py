from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


class StandardData:
    """Standard data structure returned by loading functions such as
    `pyxalign.io.loaders.pear.load_data_from_pear_format` or
    `pyxalign.io.loaders.xrf.load_data_from_xrf_format`.

    Use `launch_standard_data_viewer` to view the data interactively::

        gui = pyxalign.gui.launch_standard_data_viewer(standard_data)

    """

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
        """Initialize the StandardData object with projection data and
        metadata.

        Args:
            projections (dict[int, np.ndarray]): Mapping of scan 
                numbers to projection arrays.
            angles (np.ndarray): Array of angles corresponding to each 
                scan.
            scan_numbers (np.ndarray): Array of scan numbers.
            file_paths (Optional[dict]): Optional mapping of scan 
                numbers to file paths.
            probe_positions (Optional[dict[int, np.ndarray]]): Optional
                mapping of scan numbers to probe positions.
            probe (Optional[np.ndarray]): Optional probe data.
            pixel_size (Optional[float]): Optional pixel size for the 
                projections.

        """
        self.projections = projections
        self.angles = angles
        self.scan_numbers = scan_numbers
        self.file_paths = file_paths
        self.probe_positions = probe_positions
        self.probe = probe
        self.pixel_size = pixel_size

        # Force all angles to be in a 360 degree range
        # I like to keep values similar to raw data when possible, so I only
        # apply mod if the range is outside the 360 degree range
        if np.max(self.angles) - np.min(self.angles):
            self.angles %= 360

    def drop_scans(self, scan_numbers_to_drop: list[int]):
        """Remove specified scans from the data.

        Args:
            scan_numbers_to_drop (list[int]): List of scan numbers to be
            removed.

        """
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
        """Plot a sample projection for a given index.

        Args:
            index (int): Index of the scan to plot. Defaults to 0.

        """
        scan_number = list(self.scan_numbers)[index]
        plt.title(f"Scan {scan_number}")
        plt.imshow(np.angle(self.projections[scan_number]), cmap="bone")
        plt.show()

    def get_minimum_size_for_projection_array(self) -> np.ndarray:
        """Calculate the minimum size for the projection array.

        Returns:
            np.ndarray: Minimum size of the projection array as an 
                array of two integers.

        """
        return np.array(
            (
                np.max([v.shape[0] for v in self.projections.values()]),
                np.max([v.shape[1] for v in self.projections.values()]),
            )
        ).astype(int)
