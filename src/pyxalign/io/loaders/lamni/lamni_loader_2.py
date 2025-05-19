import numpy as np
import os
from scipy.io import loadmat
from pyxalign.io.loaders.lamni.base_loader import BaseLoader
from pyxalign.io.loaders.lamni.base_loader import generate_single_projection_sub_folder
from pyxalign.timing.timer_utils import timer, InlineTimer


class LamniLoaderVersion2(BaseLoader):
    analysis_folders: dict[int, list[str]] = {}

    @timer()
    def get_projections_folders_and_file_names(self):
        """
        Generate the folder path for all projections and get a list of
        the files in that folder.
        """
        for scan_number in self.scan_numbers:
            proj_relative_folder_path = generate_single_projection_sub_folder(
                scan_number, n_digits=5
            )
            projection_folder = os.path.join(
                self.parent_projections_folder, proj_relative_folder_path
            )
            self.record_projection_path_and_files(projection_folder, scan_number)
        print(
            f"{len(self.projection_folders)} scans have one or more projection files.",
            flush=True,
        )

    @timer()
    def record_projection_path_and_files(self, folder: str, scan_number: int):
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

    def get_nested_analysis_folders(self, folder: str, scan_number: int, rel_path: str = ""):
        """
        Get the relative paths of all folders in the scan directory that
        contain projection files by recursing through nested folders.
        """
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
                self.get_nested_analysis_folders(full_path, scan_number)

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
            self.probe_positions[scan_number] = load_positions_from_mat_file(
                self.selected_projection_file_paths[scan_number]
            )[:, ::-1]

    def load_probe(self):
        # I assume all probes are similar, and I just load the first scan's probe
        probe = load_probe_from_mat_file(self.selected_projection_file_paths[self.scan_numbers[0]])
        # different files have different formats for the probe -- try to automatically format
        # the probe file. 
        if probe.ndim == 2 and probe.shape[1] == 1:
            probe = np.array([mode[0] for mode in probe])
            self.probe = (np.abs(probe) ** 2).sum(0)
        elif probe.ndim == 4:
            # Use only the first opr mode (axis 3) and sum over incoherent modes (axis 2)
            self.probe = (np.abs(probe[:, :, :, 0]) ** 2).sum(2).transpose()

    def load_projection_params(self):
        self.pixel_size = load_params_from_mat_file(
            self.selected_projection_file_paths[self.scan_numbers[0]]
        )


def load_positions_from_mat_file(file_path):
    mat_file_contents = loadmat(file_path, variable_names="outputs")
    return mat_file_contents["outputs"]["probe_positions"][0][0]


def load_probe_from_mat_file(file_path):
    mat_file_contents = loadmat(file_path, variable_names="probe")
    return mat_file_contents["probe"]


def load_params_from_mat_file(file_path):
    data = loadmat(file_path, variable_names=["p"])
    pixel_size = data["p"]["dx_spec"][0][0][0][0]
    return pixel_size
