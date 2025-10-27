import os
import re
from typing import Optional, TypeVar, Union
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from pyxalign.io.file_readers.mda import MDAFile, convert_extra_PVs_to_dict
from pyxalign.io.loaders.pear.options import LoaderType
from pyxalign.io.loaders.pear.options import LYNXLoadOptions, MDAPEARLoadOptions, PEARLoadOptions
from pyxalign.io.loaders.maps import get_loader_class_by_enum
from pyxalign.io.loaders.utils import generate_input_user_prompt
from pyxalign.api.types import r_type
from pyxalign.io.loaders.xrf.utils import get_scan_file_dict
from pyxalign.timing.timer_utils import timer
from pyxalign.io.loaders.pear.base_loader import BaseLoader


def get_experiment_subsets(
    parent_projections_folder: str,
    scan_numbers: np.ndarray,
    angles: np.ndarray,
    experiment_names: list[str],
    sequences: np.ndarray,
    loader_type: LoaderType,
) -> dict[str, BaseLoader]:
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
    loader_type: LoaderType,
    use_experiment_name: Optional[str] = None,
    use_sequence: Optional[str] = None,
    is_tile_scan: Optional[bool] = False,
    selected_tile: Optional[int] = None,
    select_all_by_default: bool = False,
) -> BaseLoader:
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

    if is_tile_scan:
        selected_experiment.select_tile(selected_tile)

    # Select sequences subset to load
    unique_sequences = np.unique(selected_experiment.sequences)
    # Generate description strings for each option
    options_info_list = []
    for i in range(len(unique_sequences)):
        sequence_idx = unique_sequences[i] == selected_experiment.sequences
        scan_start = selected_experiment.scan_numbers[sequence_idx].min()
        scan_end = selected_experiment.scan_numbers[sequence_idx].max()
        options_info_list += [f"scans {scan_start}-{scan_end}, {sum(sequence_idx)} scans"]
    _, selected_sequences = generate_input_user_prompt(
        load_object_type_string="sequences",
        options_list=unique_sequences,
        allow_multiple_selections=True,
        options_info_list=options_info_list,
        use_option=use_sequence,
        prepend_option_with="Sequence",
        select_all_by_default=select_all_by_default,
    )
    # selected_experiment.remove_sequences(unique_sequences[selected_sequence_idx])
    selected_experiment.remove_sequences(sequences_to_keep=selected_sequences)

    return subsets[selected_experiment_name]


@timer()
def extract_info_from_lamni_dat_file(
    dat_file_path: str, scan_start: Optional[int] = None, scan_end: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Extract scan number, measurement angle, and experiment name from the
    dat file included with lamni experiments
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
    dat_file_contents["experiment_name"] = dat_file_contents["experiment_name"].fillna("")
    if scan_start is not None:
        idx = dat_file_contents["scan_number"] >= scan_start
        dat_file_contents = dat_file_contents[idx]
    if scan_end is not None:
        idx = dat_file_contents["scan_number"] <= scan_end
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


@timer()
def load_experiment(
    parent_projections_folder: str,
    n_processes: int,
    options: PEARLoadOptions,
) -> BaseLoader:
    """
    Load an experiment that is saved with the lamni structure.
    """
    scan_numbers, angles, experiment_names, sequences = extract_experiment_info(options)
    # If there is only one experiment name found, automatically select that one
    if len(np.unique(experiment_names)) == 1:
        # options.base.selected_experiment_name = experiment_names[0]
        selected_experiment_name = experiment_names[0]
    else:
        selected_experiment_name = options.selected_experiment_name
    # If there is only one sequence found, automatically select that one
    if len(np.unique(sequences)) == 1:
        selected_sequences = [sequences[0]]
    else:
        selected_sequences = options.selected_sequences
        # options.base.selected_sequences = [sequences[0]]

    if isinstance(options, LYNXLoadOptions):
        is_tile_scan = options.is_tile_scan
        selected_tile = options.selected_tile
    else:
        is_tile_scan = False
        selected_tile = None

    selected_experiment = select_experiment_and_sequences(
        parent_projections_folder,
        scan_numbers,
        angles,
        experiment_names,
        sequences,
        options.base.loader_type,
        use_experiment_name=selected_experiment_name,
        use_sequence=selected_sequences,
        is_tile_scan=is_tile_scan,
        selected_tile=selected_tile,
        select_all_by_default=options.base.select_all_by_default,
    )
    # Get paths to all existing projection files for the given scan numbers
    selected_experiment.get_projections_folders_and_file_names(options.base.file_pattern)
    # Extract the unique file string for all projection files, and filter
    # out the ones that don't match user specified inputs
    selected_experiment.get_matching_ptycho_file_strings(
        options.base.only_include_files_with,
        options.base.exclude_files_with,
        options.base.file_pattern_priority_list,
        options.base.skip_files_not_in_priority_list,
    )
    selected_experiment.select_projections(
        options.base.selected_ptycho_strings,
        options.base.ask_for_backup_files,
        options.base.select_all_by_default,
    )

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
    options: PEARLoadOptions,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    if isinstance(options, LYNXLoadOptions):
        scan_numbers, angles, experiment_names, sequences = extract_info_from_lamni_dat_file(
            options.dat_file_path,
            options.base.scan_start,
            options.base.scan_end,
        )
    elif isinstance(options, MDAPEARLoadOptions):
        scan_numbers, angles = extract_info_from_mda_file(
            options.mda_folder,
            options._mda_file_pattern,
            options._angle_pv_string,
            options.base.scan_start,
            options.base.scan_end,
        )
        # data of this type does not have experiment_names or sequences, so we have to
        # make dummy values
        experiment_names = [""] * len(scan_numbers)
        sequences = np.zeros(len(scan_numbers), dtype=int)

    # Filter scan numbers
    if options.base.scan_list is not None:
        keep_idx = [scan in options.base.scan_list for scan in scan_numbers]
        scan_numbers = scan_numbers[keep_idx]
        angles = angles[keep_idx]
        experiment_names = [name for i, name in enumerate(experiment_names) if keep_idx[i]]
        sequences = sequences[keep_idx]

    return scan_numbers, angles, experiment_names, sequences


def extract_info_from_mda_file(
    mda_folder: str,
    mda_file_pattern: str,
    angle_pv_string: str,
    scan_start: Optional[int] = None,
    scan_end: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the measurement angles for the selected scan numbers by
    extracting the measurement angle from the scan's mda file.

    Args:
        mda_folder (str): folder containing the mda files
        mda_file_pattern (str): Regular Expression file pattern that
            matches the file string of the mda file.
        angle_pv_string (str): the string that corresponds to the
            rotation angle entry in the extra PVs
        scan_start (str, optional): lower bound of scans to load.
            Defaults to None.
        scan_end (str, optional): upper bound of scans to load.
            Defaults to None.

    Returns:
        A tuple containing:
            - np.ndarray: an array of the loaded scan numbers
            - np.ndarray: an array of the loaded measurement angles
    """
    file_names_dict = get_scan_file_dict(os.listdir(mda_folder), mda_file_pattern)
    angles = np.array([], dtype=r_type)
    scan_numbers = np.array([], dtype=int)
    for current_scan, current_file in file_names_dict.items():
        above_lower_bound = scan_start is None or current_scan > scan_start
        below_upper_bound = scan_end is None or current_scan < scan_end
        if not above_lower_bound or not below_upper_bound:
            continue
        mda_file_path = os.path.join(mda_folder, current_file)
        try:
            mda_file = MDAFile.read(Path(mda_file_path))
            pv_dict = convert_extra_PVs_to_dict(mda_file)
            angles = np.append(angles, pv_dict[angle_pv_string].value[0])
            scan_numbers = np.append(scan_numbers, current_scan)
        except Exception:
            print(
                f"An error occured when attempting to read scan file {current_file}, skipping file."
            )

    # lamino_angle = pv_dict["2xfm:m12.VAL"].value[0]

    # this is a band-aid fix for now, because this is not the correct spot
    # to put this conversion
    angles[:] = -angles

    return scan_numbers, angles