from typing import Union
import numpy as np
import os
from pyxalign.io.loaders.pear.base_loader import BaseLoader
from pyxalign.io.loaders.pear.base_loader import generate_single_projection_sub_folder
from pyxalign.timing.timer_utils import timer, InlineTimer
from pyxalign.io.loaders.utils import count_digits, extract_s_digit_strings
from abc import ABC
from pathlib import Path


class NestedLoader(BaseLoader, ABC):
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
    def find_paths_with_glob(
        self, folder: str, scan_number: int, pattern: str
    ) -> Union[list[str], None]:
        if os.path.exists(folder):
            full_paths = list(Path(folder).glob(pattern))
            return full_paths

    @timer()
    def record_projection_path_and_files(self, folder: str, scan_number: int, pattern: str):
        """Get full file paths to all existing projection files for the given scan number"""
        # duplicates pearloader

        # If glob pattern is specified, only use that to find files
        if pattern is not None:
            full_paths = self.find_paths_with_glob(folder, scan_number, pattern)
            if full_paths is not None:
                self.projection_folders[scan_number] = folder
                self.available_projection_files[scan_number] = [
                    os.path.relpath(path, folder) for path in full_paths
                ]
            return

        # Get all existing projection paths
        if os.path.exists(folder) and os.listdir(folder) != []:
            # Record the absolute path to the scan folder containing ptycho results
            self.projection_folders[scan_number] = folder
            # Initialize list of analysis folders for this scan. Each entry of the
            # list will be a string with with some name like "Ndp128_LSQML_c1234_m0.5_gaussian_p5_mm_opr2_ic"
            self.analysis_folders[scan_number] = []
            inline_timer = InlineTimer("get_nested_analysis_folders")
            inline_timer.start()
            # Populate self.analysis_folders[scan_number] with the list of
            # ptycho results subfolder names
            self.get_nested_analysis_folders(folder, scan_number)
            inline_timer.end()
            # Initialize available_projection_files, which will contain the full
            # path to all available projection files for this scan
            self.available_projection_files[scan_number] = []
            for analysis_sub_folder in self.analysis_folders[scan_number]:
                # Get list of all files in the ptycho results sub-folder
                file_names = os.listdir(os.path.join(folder, analysis_sub_folder))
                # Add file names that are in the expected projection file format to
                # the list
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
    def check_if_projection_file(file_path: str) -> bool:
        raise NotImplementedError
