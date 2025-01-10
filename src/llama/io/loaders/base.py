from abc import ABC
from typing import Optional
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
        pass

    def select_experiment(self) -> ExperimentSubset:
        _, selected_key = generate_selection_user_prompt(
            load_object_type_string="experiment",
            options_list=self.subsets.keys(),
            options_info_list=[subset.n_scans for subset in self.subsets.values()],
            options_info_type_string="scans",
        )
        return self.subsets[selected_key]

    def get_experiment_subsets(self):
        raise NotImplementedError


def generate_experiment_description(
    option_string: str,
    index: int,
    options_info_list: Optional[list[str]] = None,
    options_info_type_string: Optional[str] = None,
):
    experiment_string = f"{index+1}. {option_string} ("
    if options_info_list is not None:
        experiment_string += f"{options_info_list[index]}"
        if options_info_type_string is not None:
            experiment_string += f" {options_info_type_string}"
    return experiment_string + ")\n"


def generate_selection_user_prompt(
    load_object_type_string: str,
    options_list: list[str],
    options_info_list: Optional[list[str]] = None,
    options_info_type_string: Optional[str] = None,
) -> tuple[int, str]:
    # Ensure inputs are lists
    options_list = list(options_list)
    if options_info_list is not None:
        options_info_list = list(options_info_list)
    # Generate the user prompt
    prompt = f"Select the {load_object_type_string} to load:\n"
    for index, option_string in enumerate(options_list):
        prompt += generate_experiment_description(
            option_string, index, options_info_list, options_info_type_string
        )
    # Prompt the user to make a selection
    allowed_inputs = range(0, len(options_list))
    while True:
        try:
            user_input = input(prompt)
            input_index = int(user_input) - 1
            if input_index not in allowed_inputs:
                raise ValueError
            else:
                print(f"Selected option {input_index + 1}. {options_list[input_index]}")
                return (input_index, options_list[input_index])
        except ValueError:
            print(f"Invalid input. Please enter a number 1 through {allowed_inputs[-1]}.")


if __name__ == "__main__":
    test_dict = {"option_1": "a", "option_2": "b", "option_3": "c"}
    result = generate_selection_user_prompt(
        load_object_type_string="experiment",
        options_list=test_dict.keys(),
        options_info_list=test_dict.values(),
        options_info_type_string="test_type_string",
    )
    print("returned: ", result)
