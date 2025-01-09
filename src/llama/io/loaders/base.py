from abc import ABC
from ast import Name
import os
from tkinter import N
import pandas as pd
import numpy as np


class ExperimentSubset:
    def __init__(
        self,
        scan_numbers: np.ndarray,
        angles: np.ndarray,
        experiment_name: str,
        parent_projections_folder: str,
    ):
        self.angles = angles
        self.scan_numbers = scan_numbers
        self.experiment_name = experiment_name
        self.parent_projections_folder = parent_projections_folder
        self.projection_folder: list[str] = []
        self.projection_files: list[str] = []

    def get_projection_analysis_file_info(self, *args, **kwargs):
        # Get the folders containing and names of the projection files for each scan number
        raise NotImplementedError

    def record_projection_path_and_files(self, *args, **kwargs):
        # Save a list of projection folder and file names for each scan number
        raise NotImplementedError


class ExperimentLoader(ABC):
    scan_numbers: np.ndarray
    angles: np.ndarray
    experiment_names: list[str]
    subsets: dict[str, ExperimentSubset]
    selected_experiment: ExperimentSubset
    parent_projections_folder: str
    experiment_subset_class: type

    def load_experiment(self):
        raise NotImplementedError

    def get_basic_experiment_metadata(self, *args, **kwargs):
        raise NotImplementedError

    def get_projections(self, *args, **kwargs):
        raise NotImplementedError

    def select_experiment(self) -> ExperimentSubset:
        def generate_experiment_description(experiment_name: str, i: int):
            n_scans = len(self.subsets[experiment_name].scan_numbers)
            experiment_string = f"{i+1}. {experiment_name} ({n_scans} scans)\n"
            return experiment_string

        prompt = "Select the experiment to load:\n"
        for index, name in enumerate(self.subsets.keys()):
            prompt += generate_experiment_description(name, index)
        allowed_inputs = range(1, len(self.subsets) + 1)
        while True:
            try:
                # user_input = input(prompt)
                user_input = 1
                input_index = int(user_input)
                if input_index not in allowed_inputs:
                    raise ValueError
                else:
                    return list(self.subsets.values())[input_index - 1]
            except ValueError:
                print(f"Invalid input. Please enter a number 1 through {allowed_inputs[-1]}.")

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


def txt_to_dataframe(file_path: str, column_names: list[str], delimiter: str, header=None):
    try:
        # Read the space-delimited file into a DataFrame
        df = pd.read_csv(file_path, names=column_names, delimiter=delimiter, header=header)
        return df
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


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


def generate_single_projection_sub_folder(scan_number: int, n_digits) -> str:
    "Generate name of subfolder corresponding to a single projection"
    return f"S{str(scan_number).zfill(n_digits)}"


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
