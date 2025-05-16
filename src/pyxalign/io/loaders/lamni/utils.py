import os
import re
from typing import Optional
import pandas as pd
import numpy as np
from pyxalign.io.loaders.enums import ExperimentInfoSourceType, LamniLoaderType
from pyxalign.io.loaders.lamni.options import LamniLoadOptions
from pyxalign.io.loaders.maps import get_loader_class_by_enum, LoaderInstanceType
from pyxalign.io.loaders.utils import generate_input_user_prompt
from pyxalign.api.types import r_type
from pyxalign.timing.timer_utils import InlineTimer, timer


def get_experiment_subsets(
    parent_projections_folder: str,
    scan_numbers: np.ndarray,
    angles: np.ndarray,
    experiment_names: list[str],
    sequences: np.ndarray,
    loader_type: LamniLoaderType,
) -> dict[str, LoaderInstanceType]:
    subsets = {}
    for unique_name in np.unique(experiment_names):
        idx = [index for index, name in enumerate(experiment_names) if name == unique_name]
        loader = get_loader_class_by_enum(loader_type)
        subsets[unique_name] = loader(
            scan_numbers[idx],
            angles[idx],
            unique_name,
            sequences[idx],
            parent_projections_folder,
        )
    return subsets


@timer()
def select_experiment_and_sequences(
    parent_projections_folder: str,
    scan_numbers: np.ndarray,
    angles: np.ndarray,
    experiment_names: list[str],
    sequences: np.ndarray,
    loader_type: LamniLoaderType,
    use_experiment_name: Optional[str] = None,
    use_sequence: Optional[str] = None,
) -> LoaderInstanceType:
    """
    Select the experiment you want to load.
    """
    # Select experiment subset to load
    subsets = get_experiment_subsets(
        parent_projections_folder, scan_numbers, angles, experiment_names, sequences, loader_type
    )
    _, selected_experiment_name = generate_input_user_prompt(
        load_object_type_string="experiment",
        options_list=subsets.keys(),
        options_info_list=[subset.n_scans for subset in subsets.values()],
        options_info_type_string="scans",
        use_option=use_experiment_name,
    )
    selected_experiment = subsets[selected_experiment_name]

    # Select sequences subset to load
    # Generate description strings for each option
    unique_sequences = np.unique(selected_experiment.sequences)
    options_info_list = []
    for i in range(len(unique_sequences)):
        sequence_idx = unique_sequences[i] == selected_experiment.sequences
        scan_start = selected_experiment.scan_numbers[sequence_idx].min()
        scan_end = selected_experiment.scan_numbers[sequence_idx].max()
        options_info_list += [f"scans {scan_start}-{scan_end}, {sum(sequence_idx)} scans"]
    selected_sequence_idx, _ = generate_input_user_prompt(
        load_object_type_string="sequences",
        options_list=unique_sequences,
        allow_multiple_selections=True,
        options_info_list=options_info_list,
        use_option=use_sequence,
        prepend_option_with="Sequence",
    )
    selected_experiment.remove_sequences(unique_sequences[selected_sequence_idx])

    return subsets[selected_experiment_name]


@timer()
def extract_info_from_lamni_dat_file(
    dat_file_path: str,
    scan_start: Optional[int] = None,
    scan_end: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
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
    if scan_start is not None:
        idx = dat_file_contents["scan_number"] > scan_start
        dat_file_contents = dat_file_contents[idx]
    if scan_end is not None:
        idx = dat_file_contents["scan_number"] < scan_end
        dat_file_contents = dat_file_contents[idx]

    scan_numbers = dat_file_contents["scan_number"].to_numpy()
    angles = dat_file_contents["measured_rotation_angle"].to_numpy()
    experiment_names = dat_file_contents["experiment_name"].to_list()
    sequence_number = dat_file_contents["sequence"].to_numpy()

    # remove duplicates if they are found
    if len(np.unique(scan_numbers)) != len(scan_numbers):
        print(
            "WARNING: duplicate scan numbers found in experiment file. "
            + "Only the first duplicate will be kept."
        )
        scan_numbers_no_duplicates = []
        keep_idx = []
        for i, scan in enumerate(scan_numbers):
            if scan not in scan_numbers_no_duplicates:
                scan_numbers_no_duplicates += [scan]
                keep_idx += [i]
        scan_numbers = scan_numbers[keep_idx]
        angles = angles[keep_idx]
        experiment_names = [experiment_names[i] for i in keep_idx]
        sequence_number = sequence_number[keep_idx]

    return (scan_numbers, angles, experiment_names, sequence_number)


# def remove_duplicates(scan_numbers, angles, experiment_names, sequence_number) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:


@timer()
def load_experiment(
    dat_file_path: str,
    parent_projections_folder: str,
    n_processes: int,
    options: "LamniLoadOptions",
) -> "LoaderInstanceType":
    """
    Load an experiment that is saved with the lamni structure.
    """
    # This is the function that should be called to load data, regardless
    # of the loader being used. Re-purpose this and the loader instance's methods
    # in the future as new loaders require some changes to the structure.
    # scan_numbers, angles, experiment_names, sequences = extract_info_from_lamni_dat_file(
    #     dat_file_path, options.scan_start, options.scan_end
    # )
    scan_numbers, angles, experiment_names, sequences = extract_experiment_info(
        options, parent_projections_folder, dat_file_path
    )
    selected_experiment = select_experiment_and_sequences(
        parent_projections_folder,
        scan_numbers,
        angles,
        experiment_names,
        sequences,
        options.loader_type,
        use_experiment_name=options.selected_experiment_name,
        use_sequence=options.selected_sequences,
    )
    selected_experiment.get_projections_folders_and_file_names()
    selected_experiment.extract_metadata_from_all_titles(
        options.only_include_files_with, options.exclude_files_with
    )
    selected_experiment.select_projections(
        options.selected_metadata_list, options.ask_for_backup_files
    )
    # Print data selection settings
    print("Use these settings to bypass user-selection on next load:")
    input_settings_string = (
        f'  selected_experiment_name="{selected_experiment.experiment_name}",\n'
        + f"  selected_sequences={list(np.unique(selected_experiment.sequences))},\n"
        + f"  selected_metadata_list={insert_new_line_between_list(selected_experiment.selected_metadata_list)},\n"
    )
    if options.scan_start is not None:
        input_settings_string += f"  scan_start={options.scan_start},\n"
    if options.scan_end is not None:
        input_settings_string += f"  scan_end={options.scan_end},\n"
    input_settings_string = input_settings_string[:-1]
    print(input_settings_string, flush=True)

    # Load probe
    selected_experiment.load_probe()

    # # Load probe positions
    # selected_experiment.load_positions()

    # Load the rest of the available parameters
    selected_experiment.load_projection_params()

    # Load projections
    selected_experiment.load_projections(n_processes)

    # Load probe positions
    selected_experiment.load_positions()

    # remove scan numbers, angle for items where no
    # projection was loaded

    return selected_experiment

def extract_experiment_info(
    options: LamniLoadOptions,
    reconstructions_folder: Optional[str] = None,
    dat_file_path: Optional[str] = None,
):
    if options.scan_info_source_type == ExperimentInfoSourceType.LAMNI_DAT_FILE:
        scan_numbers, angles, experiment_names, sequences = extract_info_from_lamni_dat_file(
            dat_file_path, options.scan_start, options.scan_end
        )
    else:
        scan_numbers, angles, experiment_names, sequences = extract_info_from_folder_names(
            reconstructions_folder
        )

    return scan_numbers, angles, experiment_names, sequences


def extract_info_from_folder_names(
    folder: str,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    scan_numbers = extract_numeric_patterns(folder)
    # Implement "angle rule" later
    angles = np.zeros(len(scan_numbers), dtype=r_type) # incorrect, placeholder
    experiment_names = [""] * len(scan_numbers)
    sequences = np.zeros(len(scan_numbers), dtype=int)
    return scan_numbers, angles, experiment_names, sequences


def extract_numeric_patterns(parent_directory: str) -> np.ndarray:
    # Compile a regular expression to match folders like "S0123"
    pattern = re.compile(r"^S(\d+)$")

    extracted_numbers = []

    # List all items in the parent directory
    for item in os.listdir(parent_directory):
        # Build the full path to check if it's a directory
        full_path = os.path.join(parent_directory, item)

        # Check if the item is a directory and follows the "S####" format
        if os.path.isdir(full_path):
            match = pattern.match(item)
            if match:
                # Extract the numeric portion and convert it to an integer
                num_pattern = int(match.group(1))
                extracted_numbers.append(num_pattern)

    # Convert the list of numbers into a NumPy array
    return np.array(extracted_numbers)

def insert_new_line_between_list(list_of_strings: list[str]):
    return "[\n " + ",\n ".join(f'"{item}"' for item in list_of_strings) + "\n]"
