import multiprocessing as mp
from time import time
import re
from typing import Callable, Optional, Union
import traceback
import h5py
import numpy as np
from scipy import stats
from tqdm import tqdm
from pyxalign.transformations.functions import image_crop_pad
from pyxalign.io.loaders.enums import LoaderType
from pyxalign.api.constants import divisor
from IPython import get_ipython

border = 60 * "-"


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


def generate_experiment_description(
    option_string: str,
    index: int,
    options_info_list: Optional[list[str]] = None,
    options_info_type_string: Optional[str] = None,
):
    experiment_string = f"   {index+1}. {option_string} ("
    if options_info_list is not None:
        experiment_string += f"{options_info_list[index]}"
        if options_info_type_string is not None:
            experiment_string += f" {options_info_type_string}"
    return experiment_string + ")\n"


def is_valid_option_provided(
    use_option: str, options_list: list, allow_multiple_selections: bool
) -> Union[tuple, None]:
    # Use pre-provided option if it was passed in
    if use_option is not None:
        try:
            if allow_multiple_selections:
                idx = [list(options_list).index(x) for x in use_option]
            else:
                idx = list(options_list).index(use_option)
            return (idx, use_option)
        except ValueError:
            print("Provided option is not allowed because it is not available in `options_list`")
            print(traceback.format_exc())


def prompt_input_processing(options_list: list, options_info_list: Optional[list[str]]):
    # Ensure inputs are lists
    options_list = list(options_list)
    if options_info_list is not None:
        options_info_list = list(options_info_list)
    return options_list, options_info_list


def get_user_input(
    options_list: list,
    prompt: str,
    allow_multiple_selections: bool,
) -> tuple[Union[int, list[int]], ...]:
    allowed_inputs = range(0, len(options_list))
    print(border + "\nUSER INPUT NEEDED\n" + prompt, flush=True)
    while True:
        try:
            # user_input = input(prompt)
            user_input = input("Enter input(s):")
            if allow_multiple_selections:
                selection_idx = [x - 1 for x in parse_space_delimited_integers(user_input)]
                is_input_allowed = np.all([idx in allowed_inputs for idx in selection_idx])
            else:
                selection_idx = int(user_input) - 1
                is_input_allowed = selection_idx in allowed_inputs
            if not is_input_allowed:
                raise ValueError
            else:
                if allow_multiple_selections:
                    # Get a list of the selected options
                    selection = [options_list[idx] for idx in selection_idx]
                    selection_string = [
                        f"{idx + 1}. {x}" for idx, x in zip(selection_idx, selection)
                    ]
                    selection_string = "  " + "\n  ".join(selection_string)
                    selection_string = f"Selected options:\n{selection_string}"
                else:
                    selection = options_list[selection_idx]
                    selection_string = f"Selected option {selection_idx + 1}. {selection}"
                print(selection_string + "\n" + border + "\n", flush=True)
                return (selection_idx, selection)
        except ValueError:
            if allow_multiple_selections:
                print(
                    f"{user_input} is not a valid input. Please enter a space delimited list "
                    + f"whose elements are between 1 and {allowed_inputs[-1] + 1}.\n",
                    flush=True,
                )
            else:
                print(
                    f"{user_input} is not a valid input. "
                    + f"Please enter a number 1 through {allowed_inputs[-1] + 1}.",
                    flush=True,
                )


def generate_input_user_prompt(
    load_object_type_string: str,
    options_list: list,
    allow_multiple_selections: bool = False,
    options_info_list: Optional[list] = None,
    options_info_type_string: Optional[str] = None,
    prepend_option_with: Optional[str] = None,
    use_option: Optional[str] = None,
    select_all_info: Optional[str] = None,
) -> tuple[int, str]:
    select_all_string = "select all"
    if select_all_info is None:
        select_all_info = ""

    provided_option = is_valid_option_provided(use_option, options_list, allow_multiple_selections)
    if provided_option is not None:
        return provided_option

    options_list, options_info_list = prompt_input_processing(options_list, options_info_list)

    # Generate the user prompt
    if allow_multiple_selections:
        prompt = (
            f"Select the {load_object_type_string} to load\n"
            + "Enter inputs as a series of integers seperated by spaces:\n"
        )
    else:
        prompt = f"Select the {load_object_type_string} to load:\n"

    if allow_multiple_selections:
        options_list = [select_all_string] + options_list
        options_info_list = [""] + options_info_list

    for index, option_string in enumerate(options_list):
        if option_string == select_all_string:
            prompt += generate_experiment_description(option_string, index)
        else:
            if prepend_option_with is not None:
                option_string = f"{prepend_option_with} {option_string}"
            prompt += generate_experiment_description(
                option_string, index, options_info_list, options_info_type_string
            )

    # Prompt the user to make a selection
    selection_idx, selection = get_user_input(
        options_list,
        prompt,
        allow_multiple_selections=allow_multiple_selections,
    )
    if allow_multiple_selections:
        # Remove "select all" entry from options list
        options_list = options_list[1:]
        # Check if "select all" in selection
        if select_all_string in selection:
            selection = options_list
            selection_idx = list(range(0, len(options_list)))
    return selection_idx, selection


def parse_space_delimited_integers(input_string: str):
    """
    Parse a space-delimited string of integers.

    This function checks if the input string is a valid space-delimited
    string of integers. If valid, it converts the string into a list of integers.
    If invalid, it returns None.

    Parameters
    ----------
    input_string : str
        The input string to validate and parse.

    Returns
    -------
    list of int or None
        A list of integers if the input string is valid, otherwise None.
    """
    parts = input_string.split()
    return [int(part) for part in parts]


def get_boolean_user_input(prompt: str) -> bool:
    print(prompt, flush=True)
    while True:
        user_input = input(f"{prompt} (y/n): ").strip().lower()
        if user_input in {"y", "yes"}:
            print("  Selected: y", flush=True)
            return True
        elif user_input in {"n", "no"}:
            print("  Selected: n", flush=True)
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.", flush=True)


def convert_projection_dict_to_array(
    projections: dict[int, np.ndarray],
    new_shape: Optional[tuple] = None,
    repair_orientation: bool = False,
    pad_mode: str = "constant",
    pad_with_mode: bool = False,
    delete_projection_dict: bool = False,

) -> np.ndarray:
    # Note: this always does some in-place replacement of the
    # passed in dict. I will fix this in a later version.
    if pad_with_mode:
        pad_mode = "constant"

    # Reorient the projections -- only needed for specific data
    # sets where some projections are 90 degrees off from where they
    # should be
    if repair_orientation:
        print("Rotating and flipping some projections...")
        target_aspect_ratio = projections[0].shape[1] / projections[0].shape[0]
        for k, projection in tqdm(projections.items()):
            aspect_ratio = projection.shape[1] / projection.shape[0]
            # Can try to change this aspect ratio if having issues later.
            reorient_this_projection = (target_aspect_ratio < 1 and aspect_ratio > 1) or (
                target_aspect_ratio > 1 and aspect_ratio < 1
            )
            if reorient_this_projection:
                print("Reorienting projection", k)
                projections[k] = np.fliplr(np.rot90(projection, -1))
        print("Rotating and flipping some projections...Completed")

    if new_shape is None:
        new_shape = np.max([projection.shape for projection in projections.values()], axis=0)
    else:
        new_shape = np.array(new_shape)

    # Force new shape to be compatible with downsampling functions with
    # downsampling up to divisor
    new_shape = (np.ceil(new_shape / (divisor * 2)) * (divisor * 2)).astype(int)
    print(f"Projection array shape: {new_shape}")

    # Initialize the projections array
    k = list(projections.keys())[0]
    projections_array = np.zeros(shape=(len(projections), *new_shape), dtype=projections[k].dtype)

    # Fix projections dimensions through cropping and padding
    print("Fixing projections dimensions...")
    # for projection in tqdm(projections.values()):
    for i, projection in tqdm(enumerate(projections.values()), total=len(projections)):
        if pad_with_mode:
            pad_value = stats.mode(np.abs(projection), axis=None).mode
        else:
            pad_value = None
        projections_array[i] = image_crop_pad(
            projection, new_shape[0], new_shape[1], pad_mode, constant_values=pad_value
        )
    print("Fixing projections dimensions...Completed")
    return projections_array


def select_loader_type_from_prompt() -> LoaderType:
    _, loader_type = generate_input_user_prompt(
        load_object_type_string="loader type",
        options_list=list(LoaderType),
    )
    return loader_type


def parallel_load_all_projections(
    file_paths: dict,
    n_processes: int,
    projection_loading_function: Callable,
) -> dict[int, np.ndarray]:
    "Use a process pool to load all of the projections"

    try:
        print("Loading projections into list...")
        t_0 = time()
        with mp.Pool(processes=n_processes) as pool:
            projections_map = tqdm(
                pool.imap(projection_loading_function, file_paths.values()), total=len(file_paths)
            )
            projections = dict(zip(file_paths.keys(), projections_map))
        print(f"Projection loading complete. Duration: {time() - t_0}")
    except Exception as ex:
        print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
        print(traceback.format_exc())

    return projections


if __name__ == "__main__":
    test_dict = {"option_1": "a", "option_2": "b", "option_3": "c"}
    result = generate_input_user_prompt(
        load_object_type_string="experiment",
        options_list=test_dict.keys(),
        options_info_list=test_dict.values(),
        options_info_type_string="test_type_string",
    )
    print("returned: ", result)


def count_digits(s):
    return len(re.findall(r"\d", s))


def extract_s_digit_strings(strings):
    pattern = r"^S\d+$"  # Matches 'S' followed by one or more digits
    return [s for s in strings if re.match(pattern, s)]
