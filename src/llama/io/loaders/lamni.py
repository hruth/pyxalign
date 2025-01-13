from typing import Optional
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
    StandardData,
)
from llama.api.types import r_type, c_type
from llama.io.loaders.utils import (
    generate_selection_user_prompt,
    get_boolean_user_input,
    load_h5_group,
)


def load_data_from_lamni_format(
    dat_file_path: str,
    parent_projections_folder: str,
    n_processes: int = 1,
    selected_experiment_name: Optional[str] = None,
    selected_metadata_list: Optional[list[str]] = None,
) -> StandardData:
    """
    Function for loading lamni-formatted projection data and returning
    it in the standardized format.
    """
    # Load lamni-formatted projection data
    lamni_loader = load_experiment(
        dat_file_path=dat_file_path,
        parent_projections_folder=parent_projections_folder,
        n_processes=n_processes,
        selected_experiment_name=selected_experiment_name,
        selected_metadata_list=selected_metadata_list,
    )
    # Get indices of scan numbers where projections were loaded
    loaded_proj_idx = [
        scan_number in lamni_loader.projections.keys() for scan_number in lamni_loader.scan_numbers
    ]
    # Load data into standard format
    standard_data = StandardData(
        lamni_loader.projections,
        lamni_loader.angles[loaded_proj_idx],
        lamni_loader.scan_numbers[loaded_proj_idx],
        lamni_loader.file_paths,
    )
    return standard_data


class LamniLoader:
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
    selected_metadata_list : list[str]
        List of projection metadata types that are allowed to be
        loaded, in prioritized order. The metadata types are strings
        extracted from the projection file names.
    projection_folders : dict[int, str]
        Dictionary that maps scan number to the path to the folder
        containing projection files.
    projection_files : dict[int, list[str]]
        Dictionary that maps scan number to files in the projection
        folder.
    projections : dict[int, np.ndarray]
        Dictionary that maps scan number to the projection array.
    file_paths : dict[int, str]
        Dictionary that maps scan number to the full projection file
        path.
    """

    ptycho_params: dict[int, dict] = {}
    selected_metadata_list: list[str] = []
    projection_folders: dict[int, str] = {}
    projection_files: dict[int, list[str]] = {}
    file_paths: dict[int, str] = {}
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

    @property
    def n_scans(self):
        return len(self.scan_numbers)

    def get_projection_analysis_file_info(self):
        """
        Find projection h5 files and record the metadata stored in the title strings
        """
        for scan_number in self.scan_numbers:
            proj_relative_folder_path = generate_projection_relative_path(
                scan_number,
                n_digits=5,
                n_scans_per_folder=1000,
            )
            projection_folder = os.path.join(
                self.parent_projections_folder, proj_relative_folder_path
            )
            self.record_projection_path_and_files(projection_folder, scan_number)
        self.extract_metadata_from_all_titles()

    def record_projection_path_and_files(self, folder: str, scan_number: int):
        self.projection_folders[scan_number] = folder
        self.projection_files[scan_number] = os.listdir(folder)

    def extract_metadata_from_all_titles(self):
        file_list = np.concatenate(list(self.projection_files.values())).ravel()
        self.unique_metadata = list(set([filter_string(file_string) for file_string in file_list]))
        # Count occurences for that metadata string
        self.metadata_count = {}
        for metadata_string in self.unique_metadata:
            self.metadata_count[metadata_string] = np.sum(
                [metadata_string in string for string in file_list]
            )

    def select_and_load_projections(
        self,
        n_processes: int,
        selected_metadata_list: Optional[None],
        ask_for_backup_metadata: bool = True,
        load_ptycho_params: bool = False,
    ):
        """
        Select which projections to load and then load the projections.
        """
        if selected_metadata_list is None:
            self.selected_metadata_list = [self.select_metadata_type()]
        else:
            self.selected_metadata_list = selected_metadata_list
        for i in range(self.n_scans):
            # Skip this iteration if there are no projection files
            if self.projection_files[self.scan_numbers[i]] == []:
                print(f"No projections loaded for {self.scan_numbers[i]}")
                continue
            while True:
                # Find file strings with matching types
                proj_file_string = self.find_matching_metadata(
                    self.selected_metadata_list, self.projection_files[self.scan_numbers[i]]
                )
                if proj_file_string is not None:
                    # get the file path to the reconstruction file
                    self.file_paths[self.scan_numbers[i]] = os.path.join(
                        self.projection_folders[self.scan_numbers[i]], proj_file_string
                    )
                    # extract the ptychography reconstruction parameters
                    if load_ptycho_params:
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
        self.projections = parallel_load_all_projections(self.file_paths, n_processes)

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


def extract_experiment_data(dat_file_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract scan number, measurement angle, and experiment name from the
    dat file
    """
    column_names = [
        "scan_number",
        "target_rotation_angle",
        "measured_rotation_angle",
        "unknown0",
        "sequence",
        "unknown1",
        "experiment_name",
    ]
    dat_file_contents = pd.read_csv(dat_file_path, names=column_names, delimiter=" ", header=None)
    dat_file_contents["experiment_name"] = dat_file_contents["experiment_name"].fillna("unlabeled")
    scan_numbers = dat_file_contents["scan_number"].to_numpy()
    angles = dat_file_contents["measured_rotation_angle"].to_numpy()
    experiment_names = dat_file_contents["experiment_name"].to_list()

    return (scan_numbers, angles, experiment_names)


def get_experiment_subsets(
    parent_projections_folder: str,
    scan_numbers: np.ndarray,
    angles: np.ndarray,
    experiment_names: list[str],
) -> dict[str, LamniLoader]:
    subsets = {}
    for unique_name in np.unique(experiment_names):
        idx = [index for index, name in enumerate(experiment_names) if name == unique_name]
        subsets[unique_name] = LamniLoader(
            scan_numbers[idx],
            angles[idx],
            unique_name,
            parent_projections_folder,
        )
    return subsets


def select_experiment(
    parent_projections_folder: str,
    scan_numbers: np.ndarray,
    angles: np.ndarray,
    experiment_names: list[str],
    use_option: Optional[str] = None,
) -> LamniLoader:
    """
    Select the experiment you want to load.
    """
    subsets = get_experiment_subsets(
        parent_projections_folder, scan_numbers, angles, experiment_names
    )
    _, selected_key = generate_selection_user_prompt(
        load_object_type_string="experiment",
        options_list=subsets.keys(),
        options_info_list=[subset.n_scans for subset in subsets.values()],
        options_info_type_string="scans",
        use_option=use_option,
    )
    return subsets[selected_key]


def load_experiment(
    dat_file_path: str,
    parent_projections_folder: str,
    n_processes: int = 1,
    selected_experiment_name: Optional[str] = None,
    selected_metadata_list: Optional[list[str]] = None,
) -> LamniLoader:
    """
    Load an experiment that is saved with the lamni structure.
    """
    scan_numbers, angles, experiment_names = extract_experiment_data(dat_file_path)
    selected_experiment = select_experiment(
        parent_projections_folder,
        scan_numbers,
        angles,
        experiment_names,
        use_option=selected_experiment_name,
    )
    selected_experiment.get_projection_analysis_file_info()
    selected_experiment.select_and_load_projections(n_processes, selected_metadata_list)

    return selected_experiment


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