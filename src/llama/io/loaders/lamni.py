from typing import Optional, Self
import numpy as np
import os
import re
import pandas as pd
import h5py
from tqdm import tqdm
import multiprocessing as mp
import traceback
from time import time
from llama.io.loaders.base import (
    ExperimentLoader,
    ExperimentSubset,
    generate_selection_user_prompt,
    get_boolean_user_input,
)
from llama.api.types import r_type, c_type


class LamniSubset(ExperimentSubset):
    ptycho_params: dict[int, dict] = {}
    selected_metadata_list: list[str] = []

    def set_selected_metadata(self, selected_metadata_list: list[str]):
        # Pass in selected metadata when you don't want to go through
        # the prompts again
        if selected_metadata_list is not None:
            self.selected_metadata_list = selected_metadata_list

    def get_projection_analysis_file_info(self):
        for scan_number in self.scan_numbers:
            proj_relative_folder_path = generate_projection_relative_path(
                scan_number,
                n_digits=5,
                n_scans_per_folder=1000,
            )
            projection_folder = os.path.join(
                self.parent_projections_folder, proj_relative_folder_path
            )
            self.record_projection_path_and_files(projection_folder)
        self.extract_metadata_from_all_titles()

    def record_projection_path_and_files(self, folder: str):
        self.projection_folders += [folder]
        self.projection_files += [os.listdir(folder)]

    def extract_metadata_from_all_titles(self):
        file_list = np.concatenate(self.projection_files).ravel()
        self.unique_metadata = list(set([filter_string(file_string) for file_string in file_list]))
        # Count occurences for that metadata string
        self.metadata_count = {}
        for metadata_string in self.unique_metadata:
            self.metadata_count[metadata_string] = np.sum(
                [metadata_string in string for string in file_list]
            )

    def select_and_load_projections(self, ask_for_backup_metadata: bool = True):
        """Select which projections to load and then load the projections."""
        if self.selected_metadata_list == []:
            self.selected_metadata_list = [self.select_metadata_type()]
        for i in range(self.n_scans):
            # Skip this iteration if there are no projection files
            if self.projection_files[i] == []:
                print(f"No projections loaded for {self.scan_numbers[i]}")
                continue
            while True:
                # Find file strings with matching types
                proj_file_string = self.find_matching_metadata(
                    self.selected_metadata_list, self.projection_files[i]
                )
                if proj_file_string is not None:
                    # get the file path to the reconstruction file
                    self.file_paths[self.scan_numbers[i]] = os.path.join(
                        self.projection_folders[i], proj_file_string
                    )
                    # extract the ptychography reconstruction parameters
                    self.ptycho_params[self.scan_numbers[i]] = load_h5_group(
                        self.file_paths[self.scan_numbers[i]],
                        "/reconstruction/p",
                    )
                    break
                elif ask_for_backup_metadata:
                    prompt = (
                        "No projection files with the specified metadata type(s) "
                        + f"were found for scan {self.scan_numbers[i]}.\n"
                        + "Select an option:\n"
                        + "y: Select another acceptable metadata type\n"
                        + "n: continue without loading\n"
                    )
                    select_new_metadata = get_boolean_user_input(prompt)
                    if select_new_metadata:
                        # Select a new metadata type to load
                        self.selected_metadata_list += [
                            self.select_metadata_type(exclude=self.selected_metadata_list)
                        ]
                    else:
                        print(f"No projections loaded for {self.scan_numbers[i]}")
                        prompt = "Remember this choice for remaining projections?"
                        ask_for_backup_metadata = not get_boolean_user_input(prompt)
                        break
                else:
                    break
        # Load projections
        self.projections = parallel_load_all_projections(self.file_paths)

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

    def load_projection_and_metadata(self, file_path: str, scan_number: int):
        self.ptycho_params[scan_number] = load_h5_group(file_path, "/reconstruction/p")
        with h5py.File(file_path) as File:
            self.projections[scan_number] = File["/reconstruction/object"][:]

    def select_metadata_type(self, exclude: Optional[list[str]] = None) -> str:
        if exclude is not None:
            remaining_metadata_options = {
                k: self.metadata_count[k] for k in self.metadata_count.keys() if k not in exclude
            }
        else:
            remaining_metadata_options = self.metadata_count

        _, selected_metadata = generate_selection_user_prompt(
            load_object_type_string="projection metadata type",
            options_list=remaining_metadata_options.keys(),
            options_info_list=[count for count in remaining_metadata_options.values()],
            options_info_type_string="scans",
        )
        return selected_metadata


class LamniLoader(ExperimentLoader):
    "Class for reading a specific file structure type"

    selected_experiment: LamniSubset
    subsets: dict[str, LamniSubset]
    experiment_subset_class: type = LamniSubset

    def __init__(self, dat_file_path: str, projections_folder: str):
        self.dat_file_path = dat_file_path
        self.parent_projections_folder = projections_folder

    def load_experiment(
        self,
        n_processes: int = 1,
        selected_experiment_name: Optional[str] = None,
        selected_metadata_list: Optional[list[str]] = None,
    ):
        self.get_basic_experiment_metadata(self.dat_file_path)
        self.selected_experiment = self.select_experiment(use_option=selected_experiment_name)
        self.selected_experiment.set_selected_metadata(selected_metadata_list)
        self.selected_experiment.get_projection_analysis_file_info()
        self.selected_experiment.select_and_load_projections()

    def get_basic_experiment_metadata(self, dat_file_path: str):
        # read dat-file
        column_names = [
            "scan_number",
            "target_rotation_angle",
            "measured_rotation_angle",
            "unknown0",
            "sequence",
            "unknown1",
            "experiment_name",
        ]
        dat_file_contents = txt_to_dataframe(
            dat_file_path, column_names, delimiter=" ", header=None
        )
        dat_file_contents["experiment_name"] = dat_file_contents["experiment_name"].fillna(
            "unlabeled"
        )

        self.scan_numbers = dat_file_contents["scan_number"].to_numpy()
        self.angles = dat_file_contents["measured_rotation_angle"].to_numpy()
        self.experiment_names = dat_file_contents["experiment_name"].to_list()

        self.get_experiment_subsets()

    def get_experiment_subsets(self):
        self.subsets = {}
        for unique_name in np.unique(self.experiment_names):
            idx = [index for index, name in enumerate(self.experiment_names) if name == unique_name]
            self.subsets[unique_name] = self.experiment_subset_class(
                self.scan_numbers[idx],
                self.angles[idx],
                unique_name,
                self.parent_projections_folder,
            )

    def get_projections(self) -> list[np.ndarray]:
        pass


def txt_to_dataframe(file_path: str, column_names: list[str], delimiter: str, header=None):
    try:
        # Read the space-delimited file into a DataFrame
        df = pd.read_csv(file_path, names=column_names, delimiter=delimiter, header=header)
        return df
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def load_h5_group(file_path, group_path="/"):
    """
    Load and print the structure and data of a group in an HDF5 file.

    :param file_path: Path to the HDF5 file.
    :param group_path: Path to the group in the HDF5 file (default is root).
    :return: A dictionary representing the group structure and data.
    """

    def recursive_load(group):
        group_data = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                # Recursively load nested groups
                group_data[key] = recursive_load(item)
            elif isinstance(item, h5py.Dataset):
                # Load dataset
                group_data[key] = item[()]  # Load the data
            else:
                print(f"Unsupported HDF5 item: {key}")
        return group_data

    with h5py.File(file_path, "r") as h5_file:
        target_group = h5_file[group_path]
        return recursive_load(target_group)


def generate_projection_relative_path(
    scan_number: int, n_digits: int, n_scans_per_folder: int
) -> str:
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


def generate_single_projection_sub_folder(scan_number: int, n_digits) -> str:
    "Generate name of subfolder corresponding to a single projection"
    return f"S{str(scan_number).zfill(n_digits)}"


def filter_string(input_string: str) -> str:
    # Define the pattern to match "S" followed by 5 digits
    pattern = re.compile(r"S\d{5}_")
    # Remove all occurrences of the pattern
    result = pattern.sub("", input_string)
    # Remove the specific string "recons.h5"
    result = result.replace("_recons.h5", "")
    # Return the cleaned string, stripping extra whitespace
    return result.strip()


def load_projection(file_path: str) -> np.ndarray:
    "Load a single projection"
    h5 = h5py.File(file_path, "r")
    projection = h5["/reconstruction/object"][:, :].astype(c_type)
    return projection


def parallel_load_all_projections(
    file_paths: dict,
    n_processes: int,
) -> dict[int, np.ndarray]:
    "Use a process pool to load all of the projections"

    try:
        print("Loading projections into list...")
        t_0 = time()
        with mp.Pool(processes=n_processes) as pool:
            projections_map = tqdm(
                pool.imap(load_projection, file_paths.values()), total=len(file_paths)
            )
            # projections_map = pool.map(load_projection, file_paths.values())
            projections = dict(zip(file_paths.keys(), projections_map))
        print(f"Loading complete. Duration: {time() - t_0}")
    except Exception as ex:
        print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
        print(traceback.format_exc())

    return projections
