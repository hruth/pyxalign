from uu import Error
import numpy as np
import os
import re
import pandas as pd
import h5py
from llama.io.loaders.base import (
    ExperimentLoader,
    ExperimentSubset,
)


class LamniSubset(ExperimentSubset):
    ptycho_params: dict[int, dict] = {}

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

    def load_projections(self):
        selected_metadata = self.select_metadata_type()
        for i in range(self.n_scans):
            if self.projection_files[i] == []:
                continue
            matched_strings = [
                string for string in self.projection_files[i] if selected_metadata in string
            ]
            if len(matched_strings) > 1:
                raise Error("More than one match obtained!")
            elif matched_strings == []:
                print(f"No projections loaded for scan {self.scan_numbers[i]}")
                continue
                # Add logic for selecting other metadata type if nothing found
                # input(prompt)
            file_path = os.path.join(self.projection_folders[i], matched_strings[0])
            self.ptycho_params[self.scan_numbers[i]]
            # self.ptycho_params[self.scan_numbers[i]] = load_h5_group(file_path, "/reconstruction/p")
            # with h5py.File(file_path) as File:
                # self.projections[self.scan_numbers[i]] = File["/reconstruction/object"][:]

    def select_metadata_type(self) -> str:
        def generate_metadata_description(metadata_string: str, i: int):
            counts = self.metadata_count[metadata_string]
            return f"{i+1}. {metadata_string} ({counts} scans)\n"

        prompt = "Select the metadata type to load:\n"
        for index, name in enumerate(self.metadata_count.keys()):
            prompt += generate_metadata_description(name, index)

        allowed_inputs = range(1, len(self.metadata_count) + 1)
        while True:
            try:
                user_input = input(prompt)
                input_index = int(user_input)
                if input_index not in allowed_inputs:
                    raise ValueError
                else:
                    return list(self.metadata_count.keys())[input_index - 1]
            except ValueError:
                print(f"Invalid input. Please enter a number 1 through {allowed_inputs[-1]}.")


class LamniLoader(ExperimentLoader):
    "Class for reading a specific file structure type"

    experiment_subset_class: type = LamniSubset

    def __init__(self, dat_file_path: str, projections_folder: str):
        self.dat_file_path = dat_file_path
        self.parent_projections_folder = projections_folder

    def load_experiment(self):
        self.get_basic_experiment_metadata(self.dat_file_path)
        self.selected_experiment = self.select_experiment()
        self.selected_experiment.get_projection_analysis_file_info()
        self.selected_experiment.load_projections()
        # self.selected_experiment.select_metadata_type()

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



