import os
import numpy as np
import re
from pyxalign.io.loaders.lamni.lamni_loader_3 import LamniLoaderVersion3
from pyxalign.io.loaders.lamni.utils import select_experiment_and_sequences, insert_new_line_between_list
from pyxalign.io.loaders.maps import get_loader_class_by_enum
from pyxalign.io.loaders.xrf.options import XRFV2LoadOptions
from pyxalign.timing.timer_utils import timer
from pyxalign.api.types import r_type

def load_xrf_experiment_v2():
    pass

@timer()
def load_experiment_xrf_v2(
    parent_projections_folder: str,
    n_processes: int,
    options: XRFV2LoadOptions,
):# -> LoaderInstanceType:
    """
    Load an experiment that is saved with the lamni structure.
    """
    # This is the function that should be called to load data, regardless
    # of the loader being used. Re-purpose this and the loader instance's methods
    # in the future as new loaders require some changes to the structure.

    # scan_numbers, angles, experiment_names, sequences = extract_experiment_data(
    #     dat_file_path, options.scan_start, options.scan_end
    # )
    scan_numbers, angles, experiment_names, sequences = (
        extract_xrf_experiment_data_from_recon_folder(parent_projections_folder)
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


# def extract_xrf_experiment_data_from_recon_folder(
#     folder: str,
# ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
#     scan_numbers = extract_numeric_patterns(folder)
#     # Implement "angle rule" later
#     angles = np.linspace(0, 360, len(scan_numbers), dtype=r_type) # incorrect, placeholder  # noqa: F821
#     experiment_names = [""] * len(scan_numbers)
#     sequences = np.zeros(len(scan_numbers), dtype=r_type)
#     return scan_numbers, angles, experiment_names, sequences


# def extract_numeric_patterns(parent_directory: str) -> np.ndarray:
#     # Compile a regular expression to match folders like "S0123"
#     pattern = re.compile(r"^S(\d+)$")

#     extracted_numbers = []

#     # List all items in the parent directory
#     for item in os.listdir(parent_directory):
#         # Build the full path to check if it's a directory
#         full_path = os.path.join(parent_directory, item)

#         # Check if the item is a directory and follows the "S####" format
#         if os.path.isdir(full_path):
#             match = pattern.match(item)
#             if match:
#                 # Extract the numeric portion and convert it to an integer
#                 num_pattern = int(match.group(1))
#                 extracted_numbers.append(num_pattern)

#     # Convert the list of numbers into a NumPy array
#     return np.array(extracted_numbers)


class XRFLoaderVersion2(LamniLoaderVersion3):
    pass


if __name__ == "__main__":
    recons_folder = "/net/micdata/data1/2ide/2025-1/Lamni-6/ptychi_recons/"
    data_out = load_experiment_xrf_v2(recons_folder, 1, XRFV2LoadOptions(only_include_files_with=["recon_Niter5000"]))
    print(data_out)