import numpy as np
import os
from llama.io.loaders.base import (
    ExperimentLoader,
    ExperimentSubset,
    generate_projection_relative_path,
    txt_to_dataframe,
)

class LamniSubset(ExperimentSubset):
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
            print(projection_folder)

    def record_projection_path_and_files(self, folder: str):
        self.projection_folder += [folder]
        self.projection_files += [os.listdir(folder)]


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
