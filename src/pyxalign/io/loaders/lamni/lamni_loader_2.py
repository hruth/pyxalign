import numpy as np
from scipy.io import loadmat
from pyxalign.io.loaders.lamni.nested_loader import NestedLoader
from pyxalign.timing.timer_utils import timer


class LamniLoaderVersion2(NestedLoader):
    @staticmethod
    @timer()
    def check_if_projection_file(file_path: str) -> bool:
        if not file_path.lower().endswith(".mat"):
            return False
        else:
            return True

    @staticmethod
    def load_single_projection(file_path: str) -> np.ndarray:
        "Load a single projection"
        projection = loadmat(file_path)["object"]
        return projection

    def load_positions(self):
        self.probe_positions = {}
        for scan_number in self.scan_numbers:
            file_path = self.selected_projection_file_paths[scan_number]
            self.probe_positions[scan_number] = loadmat(
                file_path,
                variable_names="outputs",
            )["outputs"]["probe_positions"][0][0][:, ::-1]

    def load_probe(self):
        # I assume all probes are similar, and I just load the first scan's probe
        # probe = load_probe_from_mat_file(self.selected_projection_file_paths[self.scan_numbers[0]])
        file_path = self.selected_projection_file_paths[self.scan_numbers[0]]
        probe = loadmat(file_path, variable_names="probe")["probe"]
        # different files have different formats for the probe -- try to automatically format
        # the probe file.
        if probe.ndim == 2 and probe.shape[1] == 1:
            probe = np.array([mode[0] for mode in probe])
            self.probe = (np.abs(probe) ** 2).sum(0)
        elif probe.ndim == 4:
            # Use only the first opr mode (axis 3) and sum over incoherent modes (axis 2)
            self.probe = (np.abs(probe[:, :, :, 0]) ** 2).sum(2).transpose()

    def load_projection_params(self):
        file_path = self.selected_projection_file_paths[self.scan_numbers[0]]
        self.pixel_size = loadmat(file_path, variable_names=["p"])["p"]["dx_spec"][0][0][0][0]
