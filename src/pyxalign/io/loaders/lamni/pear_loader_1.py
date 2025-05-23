from typing import Optional
import numpy as np
import os
import h5py
from pyxalign.api.types import c_type
from pyxalign.io.loaders.lamni.base_loader import BaseLoader
from pyxalign.io.loaders.lamni.base_loader import generate_single_projection_sub_folder
# from pyxalign.io.loaders.lamni.utils import count_digits, extract_s_digit_strings
from pyxalign.io.loaders.utils import count_digits, extract_s_digit_strings
from pyxalign.timing.timer_utils import InlineTimer, timer


class PearLoaderVersion1(BaseLoader):
    analysis_folders: dict[int, list[str]] = {}
    n_digits: int = None

    def get_projection_sub_folder(self, scan_number: int):
        # Determine how many digits are in the folder strings
        if self.n_digits is None:
            example_scan_folder = extract_s_digit_strings(
                os.listdir(self.parent_projections_folder)
            )[0]
            self.n_digits = count_digits(example_scan_folder)
        # Return folder string corresponding to this scan number
        return generate_single_projection_sub_folder(
            scan_number,
            n_digits=self.n_digits,
        )

    @timer()
    def record_projection_path_and_files(self, folder: str, scan_number: int):
        # duplicate lamniloaderv2
        # Get all projection folders
        if os.path.exists(folder) and os.listdir(folder) != []:
            self.projection_folders[scan_number] = folder
            # self.analysis_folders[scan_number] = os.listdir(folder)
            self.analysis_folders[scan_number] = []
            inline_timer = InlineTimer("get_nested_analysis_folders")
            inline_timer.start()
            self.get_nested_analysis_folders(folder, scan_number)
            inline_timer.end()
            self.available_projection_files[scan_number] = []
            for analysis_sub_folder in self.analysis_folders[scan_number]:
                file_names = os.listdir(os.path.join(folder, analysis_sub_folder))
                self.available_projection_files[scan_number] += [
                    os.path.join(analysis_sub_folder, file)
                    for file in file_names
                    if self.check_if_projection_file(
                        os.path.join(folder, analysis_sub_folder, file)
                    )
                ]

    @staticmethod
    def check_if_projection_file(file_path: str) -> bool:
        _, file_name = os.path.split(file_path)
        if file_name.lower().startswith("recon"):  # and file_name.endswith("Niter3000.h5"):
            return True
        else:
            return False

    def get_nested_analysis_folders(
        self,
        folder: str,
        scan_number: int,
        rel_path: str = "",
        max_levels: int = 1,
        current_level: int = 0,
    ):
        """
        Get the relative paths of all folders in the scan directory that
        contain projection files by recursing through nested folders.
        """
        if current_level > max_levels:
            return
        # Include this folder if it has a projection file
        folder_contains_projection_file = np.any(
            [self.check_if_projection_file(os.path.join(folder, x)) for x in os.listdir(folder)]
        )
        if folder_contains_projection_file:
            # relative_path = re.sub(self.projection_folders[scan_number], "", folder)
            relative_path = os.path.relpath(folder, self.projection_folders[scan_number])
            self.analysis_folders[scan_number] += [relative_path]

        # Look through nested folders
        for folder_entry in os.listdir(folder):
            full_path = os.path.join(folder, folder_entry)
            if os.path.isfile(full_path):
                continue
            else:
                self.get_nested_analysis_folders(
                    full_path, scan_number, max_levels=max_levels, current_level=current_level + 1
                )

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
        probe = load_probe_from_h5_file(self.selected_projection_file_paths[self.scan_numbers[0]])
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
            self.probe_positions[scan_number] = load_positions_from_h5_file(
                self.selected_projection_file_paths[scan_number]
            )

    @timer()
    def load_projection_params(self):
        self.pixel_size = load_params_from_h5_file(
            self.selected_projection_file_paths[self.scan_numbers[0]]
        )


@timer()
def load_params_from_h5_file(file_path):
    with h5py.File(file_path) as F:
        pixel_size = F["obj_pixel_size_m"][()]
    return pixel_size


@timer()
def load_probe_from_h5_file(file_path: str):
    with h5py.File(file_path) as F:
        probe = F["probe"][()]
    return probe


@timer()
def load_positions_from_h5_file(file_path: str):
    with h5py.File(file_path) as F:
        positions = F["positions_px"][()]
    return positions


def generate_projection_relative_path(
    scan_number: int, n_digits: int, n_scans_per_folder: int
) -> str:
    # Used in BaseLoader
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
