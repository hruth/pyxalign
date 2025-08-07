import numpy as np
import os
import h5py
from pyxalign.api.types import c_type
from pyxalign.io.loaders.pear.nested_loader import NestedLoader
from pyxalign.timing.timer_utils import timer


class PearLoaderVersion1(NestedLoader):
    @staticmethod
    def check_if_projection_file(file_path: str) -> bool:
        _, file_name = os.path.split(file_path)
        if file_name.lower().startswith("recon"):
            return True
        else:
            return False

    @staticmethod
    def load_single_projection(file_path: str) -> np.ndarray:
        "Load a single projection"
        h5 = h5py.File(file_path, "r")
        projection = h5["object"][()][0].astype(c_type)
        h5.close()
        return projection

    @timer()
    def load_probe(self):
        # I assume all probes are similar, and I just load the first scan's probe
        file_path = self.selected_projection_file_paths[self.scan_numbers[0]]
        with h5py.File(file_path) as F:
            probe = F["probe"][()]

        if len(probe.shape) == 4:
            # sum over incoherent modes (axis 0), and look at only the first opr mode (axis 1)
            self.probe = (np.abs(probe[:, 0]) ** 2).sum(0)
        else:
            raise ValueError(
                f"Probe has {len(probe.shape)} dims; expected 4 dims."
                + "Fix the load_probe method."
            )

    @timer()
    def load_positions(self):
        self.probe_positions = {}
        for scan_number in self.scan_numbers:
            file_path = self.selected_projection_file_paths[scan_number]
            with h5py.File(file_path) as F:
                self.probe_positions[scan_number] = F["positions_px"][()]

    @timer()
    def load_projection_params(self):
        file_path = self.selected_projection_file_paths[self.scan_numbers[0]]
        with h5py.File(file_path) as F:
            self.pixel_size = F["obj_pixel_size_m"][()]
