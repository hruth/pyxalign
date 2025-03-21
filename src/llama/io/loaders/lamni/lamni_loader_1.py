from typing import Optional
import numpy as np
import os
import h5py
from llama.api.types import c_type
from llama.io.loaders.lamni.base_loader import LamniLoader
from llama.io.loaders.lamni.base_loader import generate_single_projection_sub_folder
from llama.timing.timer_utils import InlineTimer, timer


class LamniLoaderVersion1(LamniLoader):
    def get_projections_folders_and_file_names(self):
        """
        Generate the folder path for all projections and get a list of
        the files in that folder.
        """
        for scan_number in self.scan_numbers:
            proj_relative_folder_path = generate_projection_relative_path(
                scan_number,
                n_digits=5,
                n_scans_per_folder=1000,
            )
            projection_folder = os.path.join(
                self.parent_projections_folder, proj_relative_folder_path
            )
            self.record_projection_path_and_files(projection_folder, scan_number)
        print(
            f"{len(self.projection_folders)} scans have one or more projection files.",
            flush=True,
        )

    def record_projection_path_and_files(self, folder: str, scan_number: int):
        if os.path.exists(folder) and os.listdir(folder) != []:
            self.projection_folders[scan_number] = folder
            self.available_projection_files[scan_number] = os.listdir(folder)

    @staticmethod
    def load_single_projection(file_path: str) -> np.ndarray:
        "Load a single projection"
        h5 = h5py.File(file_path, "r")
        projection = h5["/reconstruction/object"][:, :].astype(c_type)
        h5.close()
        return projection

    def load_probe(self):
        # I assume all probes are similar, and I just load the first scan's probe
        probe = load_probe_from_h5_file(self.selected_projection_file_paths[self.scan_numbers[0]])
        if len(probe.shape) == 2:
            self.probe = np.abs(probe) ** 2
        else:
            raise ValueError(
                f"Probe has {len(probe.shape)} dims; expected 2 dims."
                + "Fix the load_probe method."
            )

    def load_positions(self):
        self.probe_positions = {}
        for scan_number in self.scan_numbers:
            self.probe_positions[scan_number] = load_positions_from_h5_file(
                self.selected_projection_file_paths[scan_number]
            )
            # Remove probe array offset
            self.probe_positions[scan_number] += np.array(self.probe.shape) / 2
            # Offset by center pixel
            center_pixel = np.array(self.projections[scan_number].shape) / 2
            self.probe_positions[scan_number] -= center_pixel

    def load_projection_params(self):
        self.pixel_size = load_params_from_h5_file(
            self.selected_projection_file_paths[self.scan_numbers[0]]
        )

def load_params_from_h5_file(file_path):
    with h5py.File(file_path) as F:
        pixel_size = F["reconstruction"]["p"]["dx_spec"][()][0][0]
    return pixel_size

def load_probe_from_h5_file(file_path: str):
    with h5py.File(file_path) as F:
        probe = F["reconstruction"]["probes"][()]
    return probe


def load_positions_from_h5_file(file_path: str):
    with h5py.File(file_path) as F:
        positions = F["reconstruction"]["p"]["positions_0"][()].transpose()
    return positions

def generate_projection_relative_path(
    scan_number: int, n_digits: int, n_scans_per_folder: int
) -> str:
    # Used in LamniLoader
    return os.path.join(
        generate_projection_group_sub_folder(
            scan_number,
            n_scans_per_folder,
            n_digits,
        ),
        generate_single_projection_sub_folder(
            scan_number,
            n_digits,
        ),
    )


def generate_projection_group_sub_folder(
    scan_number: int, n_scans_per_folder: int, n_digits: int
) -> str:
    "Get name of subfolder containing folders for each scan number"
    lower_bound = int(np.floor(scan_number / n_scans_per_folder)) * n_scans_per_folder
    upper_bound = lower_bound + n_scans_per_folder
    start = str(lower_bound).zfill(n_digits)
    end = str(upper_bound - 1).zfill(n_digits)

    # Construct the pattern
    return f"S{start}-{end}"
