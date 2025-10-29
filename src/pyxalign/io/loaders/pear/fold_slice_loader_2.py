import numpy as np
from scipy.io import loadmat
import h5py
from pyxalign.io.loaders.pear.nested_loader import PEARNestedLoader
from pyxalign.api.types import c_type
from pyxalign.timing.timer_utils import timer


class FoldSliceLoaderVersion2(PEARNestedLoader):
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
        try:
            projection = loadmat(file_path)["object"]
        except NotImplementedError as ex:
            with h5py.File(file_path, "r") as F:
                projection = (F["object"]["real"] + 1j * F["object"]["imag"]).astype(c_type).transpose([1, 0])

        return projection

    def load_positions(self):
        self.probe_positions = {}
        for scan_number in self.scan_numbers:
            file_path = self.selected_projection_file_paths[scan_number]
            try:
                self.probe_positions[scan_number] = loadmat(
                    file_path,
                    variable_names="outputs",
                )["outputs"]["probe_positions"][0][0][:, ::-1]
            except NotImplementedError as ex:
                with h5py.File(file_path, "r") as F:
                    self.probe_positions[scan_number] = F["outputs"]["probe_positions"][
                        ()
                    ].transpose()[:, ::-1]

    def load_probe(self):
        # I assume all probes are similar, and I just load the first scan's probe
        file_path = self.selected_projection_file_paths[self.scan_numbers[0]]
        try:
            probe = loadmat(file_path, variable_names="probe")["probe"]
        except NotImplementedError as ex:
            with h5py.File(file_path, "r") as F:
                probe = (F["probe"]["real"] + 1j * F["probe"]["imag"]).astype(c_type)

        # different files have different formats for the probe -- try to automatically format
        # the probe file.
        if probe.ndim == 2 and probe.shape[1] == 1:
            probe = np.array([mode[0] for mode in probe])
            self.probe = (np.abs(probe) ** 2).sum(0)
        elif probe.ndim == 4:
            # Use only the first opr mode (axis 3) and sum over incoherent modes (axis 2)
            self.probe = (np.abs(probe[:, :, :, 0]) ** 2).sum(2).transpose()
        elif probe.ndim == 3:
            # assumption: first axis is incoherent mode, other axes are spatial
            self.probe = probe.sum(0)

    def load_projection_params(self):
        file_path = self.selected_projection_file_paths[self.scan_numbers[0]]
        try:
            self.pixel_size = loadmat(file_path, variable_names=["p"])["p"]["dx_spec"][0][0][0][0]
        except NotImplementedError as ex:
            with h5py.File(file_path, "r") as F:
                self.pixel_size = F["p"]["dx_spec"][()][0][0]
