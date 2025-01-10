from abc import ABC
import numpy as np


class ExperimentSubset:
    projection_folders: list[str] = []
    projection_files: list[str] = []
    projections: dict[int, np.ndarray] = {}

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

    def get_projection_analysis_file_info(self, *args, **kwargs):
        # Get the folders containing and names of the projection files for each scan number
        raise NotImplementedError

    def record_projection_path_and_files(self, *args, **kwargs):
        # Save a list of projection folder and file names for each scan number
        raise NotImplementedError

    @property
    def n_scans(self):
        return len(self.scan_numbers)


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
        raise NotImplementedError
