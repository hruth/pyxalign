from abc import ABC
from ast import Name
import os
import pandas as pd
import numpy as np

# The loader should:
# - get scan info from dat-file
# - use pattern recognition on saved data


class ExperimentSubset:
    def __init__(self, scan_numbers: np.ndarray, angles: np.ndarray, experiment_name: str):
        self.angles = angles
        self.scan_numbers = scan_numbers
        self.experiment_name = experiment_name


class ExperimentRecords(ABC):
    scan_numbers: np.ndarray
    angles: np.ndarray
    experiment_names: list[str]
    subsets: dict[str, ExperimentSubset]
    selected_experiment: ExperimentSubset
    selected_projections: list[np.ndarray]

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
        for index, name in enumerate(np.unique(self.experiment_names)):
            prompt += generate_experiment_description(name, index)
        allowed_inputs = range(1, len(self.subsets) + 1)
        while True:
            try:
                user_input = input(prompt)
                input_index = int(user_input)
                if input_index not in allowed_inputs:
                    raise ValueError
                else:
                    return list(self.subsets.values())[input_index]
            except ValueError:
                print(f"Invalid input. Please enter a number 1 through {allowed_inputs[-1]}.")

    def get_experiment_subsets(self):
        self.subsets = {}
        for unique_name in np.unique(self.experiment_names):
            idx = [index for index, name in enumerate(self.experiment_names) if name == unique_name]
            self.subsets[unique_name] = ExperimentSubset(
                self.scan_numbers[idx], self.angles[idx], unique_name
            )


class TestRecords(ExperimentRecords):
    def __init__(self, dat_file_path: str, projections_folder: str):
        self.dat_file_path = dat_file_path
        self.projections_folder = projections_folder

    def load_experiment(self):
        self.get_basic_experiment_metadata(dat_file_path)
        self.selected_experiment = self.select_experiment()
        self.selected_projections = self.get_projections()

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
