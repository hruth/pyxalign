from typing import Optional
import h5py
import numpy as np
import os
from abc import ABC
import re
from llama.io.loaders.utils import (
    border,
    generate_input_user_prompt,
    get_boolean_user_input,
    load_h5_group,
    parallel_load_all_projections,
)
from llama.timing.timer_utils import InlineTimer, timer


class LamniLoader(ABC):
    """
    Class for loading ptychography reconstructions saved in the Lamni
    file structure format.

    Parameters
    ----------
    scan_numbers : np.ndarray
        Scan number of each projection, according to the dat file.
    angles : np.ndarray
        Measurement angles for each projection, according to the dat
        file.
    experiment_name : str
        Name of the experiment to load, according to the dat file.
    parent_projections_folder : str
        Path to folder containing saved projection data.

    Attributes
    ----------
    ptycho_params : dict[int, dict]
        Stores the ptychoshelves ptychography reconstruction settings
        that are stored in the `p` group of the h5 file.
    projection_folders : dict[int, str]
        Dictionary that maps scan number to the path to the folder
        containing projection files.
    projection_files : dict[int, list[str]]
        Dictionary that maps scan number to files in the projection
        folder.
    file_paths : dict[int, str]
        Dictionary that maps scan number to the full projection file
        path.
    projections : dict[int, np.ndarray]
        Dictionary that maps scan number to the projection array.
    """

    ptycho_params: dict[int, dict] = {}
    projection_folders: dict[int, str] = {}
    available_projection_files: dict[int, list[str]] = {}
    selected_projection_file_paths: dict[int, str] = {}

    projections: dict[int, np.ndarray] = {}
    probe_positions: Optional[dict[int, np.ndarray]] = None
    probe: Optional[np.ndarray] = None
    pixel_size: Optional[float] = None

    def __init__(
        self,
        scan_numbers: np.ndarray,
        angles: np.ndarray,
        experiment_name: str,
        sequences: np.ndarray,
        parent_projections_folder: str,
    ):
        self.angles = angles
        self.scan_numbers = scan_numbers
        self.experiment_name = experiment_name
        self.sequences = sequences
        self.parent_projections_folder = parent_projections_folder

        self._post_init()

    def _post_init(self):
        "For implementing child-specific initialization code"
        pass

    @property
    def n_scans(self):
        return len(self.scan_numbers)

    @timer()
    def remove_sequences(self, sequences_to_keep: list[int]):
        keep_index = [sequence in sequences_to_keep for sequence in self.sequences]
        # Remove unwanted sequences from arrays and lists
        self.sequences = self.sequences[keep_index]
        self.angles = self.angles[keep_index]
        self.scan_numbers = self.scan_numbers[keep_index]

        # Remove data from dicts
        def return_dict_subset(d: dict, keep_keys):
            return {k: v for k, v in d.items() if k in keep_keys}

        self.projection_folders = return_dict_subset(self.projection_folders, sequences_to_keep)
        self.available_projection_files = return_dict_subset(
            self.available_projection_files, sequences_to_keep
        )
        self.selected_projection_file_paths = return_dict_subset(
            self.selected_projection_file_paths, sequences_to_keep
        )
        self.projections = return_dict_subset(self.projections, sequences_to_keep)
        self.ptycho_params = return_dict_subset(self.ptycho_params, sequences_to_keep)

    @timer()
    def extract_metadata_from_all_titles(
        self,
        only_include_files_with: Optional[list[str]] = None,
        exclude_files_with: Optional[list[str]] = None,
    ):
        file_list = np.concatenate(list(self.available_projection_files.values())).ravel()
        self.unique_metadata = list(set([filter_string(file_string) for file_string in file_list]))
        # Remove data that doesn't fit specified conditions
        self.unique_metadata = self.filter_file_list(only_include_files_with, exclude_files_with)
        # Count occurences for that metadata string
        self.metadata_count = {}
        for metadata_string in self.unique_metadata:
            self.metadata_count[metadata_string] = np.sum(
                [metadata_string in string for string in file_list]
            )

    @timer()
    def filter_file_list(self, only_include_files_with: list[str], exclude_files_with: list[str]):
        if only_include_files_with is None:
           only_include_files_with = []
        if exclude_files_with is None:
             exclude_files_with = []
        filtered_list = []
        for file_string in self.unique_metadata:
            in_include = np.all([x in file_string for x in only_include_files_with])
            not_in_exclude = np.all([x not in file_string for x in exclude_files_with])
            if in_include and not_in_exclude:
                filtered_list += [file_string]
        return filtered_list

    @timer()
    def select_projections(
        self,
        selected_metadata_list: Optional[list[str]],
        ask_for_backup_metadata: bool,
    ):
        """
        Select which projections to load.
        """
        if selected_metadata_list is None:
            self.selected_metadata_list = self.select_metadata_type()
        else:
            self.selected_metadata_list = selected_metadata_list
        for scan_number in self.projection_folders.keys():
            while True:
                # Find file strings with matching types
                proj_file_string = self.find_matching_metadata(
                    self.selected_metadata_list, self.available_projection_files[scan_number]
                )
                if proj_file_string is not None:
                    # get the file path to the reconstruction file
                    self.selected_projection_file_paths[scan_number] = os.path.join(
                        self.projection_folders[scan_number], proj_file_string
                    )
                    break
                elif ask_for_backup_metadata:
                    print(border, flush=True)
                    prompt = (
                        "No projection files with the specified metadata type(s) "
                        + f"were found for scan {scan_number}.\n"
                        + "Select an option:\n"
                        + "y: Select another acceptable metadata type\n"
                        + "n: continue without loading"
                    )
                    select_new_metadata = get_boolean_user_input(prompt)
                    if select_new_metadata:
                        # Select a new metadata type to load
                        self.selected_metadata_list += self.select_metadata_type(exclude=self.selected_metadata_list)
                    else:
                        print(f"No projections loaded for {scan_number}", flush=True)
                        prompt = "Remember this choice for remaining projections?"
                        ask_for_backup_metadata = not get_boolean_user_input(prompt)
                        print(border, flush=True)
                        break
                else:
                    break
        # Remove scan numbers and angles for projections that won't be loaded
        selected_projections_idx = [
            scan_number in self.selected_projection_file_paths.keys()
            for scan_number in self.scan_numbers
        ]
        self.angles = self.angles[selected_projections_idx]
        self.scan_numbers = self.scan_numbers[selected_projections_idx]

    @timer()
    def load_projections(self, n_processes: int):
        # Load projections
        self.projections = parallel_load_all_projections(
            self.selected_projection_file_paths, n_processes, self.load_single_projection
        )

    @timer()
    def find_matching_metadata(
        self, selected_metadata_list: list[str], projection_files: list[str]
    ) -> str:
        for selected_metadata in selected_metadata_list:
            matched_strings = [string for string in projection_files if selected_metadata in string]
            if len(matched_strings) == 1:
                return matched_strings[0]
            elif len(matched_strings) > 1:
                # If this gets triggered, I just need to create a match
                # finder that breaks the "tie"
                raise Exception(
                    "More than one match obtained!\nIf you get this error, it"
                    + " means there is a bug in the code that needs to be fixed."
                )

    @timer()
    def load_projection_and_metadata(self, file_path: str, scan_number: int):
        self.ptycho_params[scan_number] = load_h5_group(file_path, "/reconstruction/p")
        with h5py.File(file_path) as File:
            self.projections[scan_number] = File["/reconstruction/object"][:]

    @timer()
    def select_metadata_type(self, exclude: Optional[list[str]] = None) -> list[str]:
        if exclude is not None:
            remaining_metadata_options = {
                k: self.metadata_count[k] for k in self.metadata_count.keys() if k not in exclude
            }
        else:
            remaining_metadata_options = self.metadata_count

        n_scans_per_option = [count for count in remaining_metadata_options.values()]
        _, selected_metadata = generate_input_user_prompt(
            load_object_type_string="projection metadata type",
            options_list=remaining_metadata_options.keys(),
            options_info_list=n_scans_per_option,
            options_info_type_string="scans",
            allow_multiple_selections=True,
            select_all_info = np.sum(n_scans_per_option),
        )
        return selected_metadata

    @staticmethod
    def load_single_projection(self):
        raise NotImplementedError

    def load_positions(self):
        pass

    def load_probe(self):
        pass

    def load_projection_params(self):
        pass


def generate_single_projection_sub_folder(scan_number: int, n_digits) -> str:
    "Generate name of subfolder corresponding to a single projection"
    return f"S{str(scan_number).zfill(n_digits)}"


def filter_string(input_string: str) -> str:
    # Define the pattern to match "S" followed by 5 digits
    pattern = re.compile(r"S\d{5}_")
    # Remove all occurrences of the pattern
    result = pattern.sub("", input_string)
    # Remove the specific string "recons.h5"
    # result = result.replace("_recons.h5", "")
    result = result.replace(".h5", "")
    # Return the cleaned string, stripping extra whitespace
    return result.strip()
