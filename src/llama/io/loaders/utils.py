import chunk
from math import e
from typing import Optional, Union
import traceback
import h5py
import numpy as np
import time
from scipy import stats

from tqdm import tqdm

from llama.transformations.functions import image_crop_pad
from llama.api.types import c_type

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


def is_valid_option_provided(use_option: str, options_list: list, allow_multiple_selections: bool) -> Union[tuple, None]:
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
    prepend_option_with: str,
) -> tuple[Union[int, list[int]], ...]:
    allowed_inputs = range(0, len(options_list))
    print(border + "\nUSER INPUT NEEDED\n" + prompt, flush=True)
    while True:
        try:
            user_input = input(prompt)
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
                    selection = [options_list[idx] for idx in selection_idx]
                    selection_string = [
                        f"{idx + 1}. {prepend_option_with} {x}"
                        for idx, x in zip(selection_idx, selection)
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
) -> tuple[int, str]:
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
    for index, option_string in enumerate(options_list):
        if prepend_option_with is not None:
            option_string = f"{prepend_option_with} {option_string}"
        prompt += generate_experiment_description(
            option_string, index, options_info_list, options_info_type_string
        )
    # Prompt the user to make a selection
    return get_user_input(
        options_list,
        prompt,
        allow_multiple_selections=allow_multiple_selections,
        prepend_option_with=prepend_option_with,
    )


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
    divisible_by: int = 32,
    chunk_length: int = 20,
    delete_projection_dict: bool = False,
) -> np.ndarray:
    if pad_with_mode:
        pad_mode = "constant"

    # Reorient the projections -- only needed for specific data
    # sets where projections are 90 degrees off from where they
    # should be
    if repair_orientation:
        print("Rotating and flipping some projections...")
        target_aspect_ratio = projections[0].shape[1] / projections[0].shape[0]
        # for i in tqdm(range(len(projections))):
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
        new_shape = np.array([projection.shape for projection in projections.values()]).max(axis=0)
    else:
        new_shape = np.array(new_shape)

    # Force new shape to be compatible with downsampling functions with
    # downsampling up to divisible_by
    new_shape = (np.floor(new_shape / (divisible_by * 2)) * (divisible_by * 2)).astype(int)

    # Fix projections dimensions through cropping and padding
    print("Fixing projections dimensions...")
    for k, projection in tqdm(projections.items()):
        if pad_with_mode:
            pad_value = stats.mode(np.abs(projection), axis=None).mode
        else:
            pad_value = None
        projections[k] = image_crop_pad(
            projection, new_shape[0], new_shape[1], pad_mode, constant_values=pad_value
        )
    print("Fixing projections dimensions...Completed")

    # Convert to array in chunks to avoid memory issues
    print("Converting list to array...")
    n_iterations = int(np.ceil(len(projections) / chunk_length))
    all_keys = list(projections.keys())
    for i in tqdm(range(n_iterations)):
        keys = all_keys[i * chunk_length : (i + 1) * chunk_length]
        if i == 0:
            projections_array = np.stack([projections[key] for key in keys])
        else:
            projections_array = np.append(
                projections_array, np.stack([projections[key] for key in keys]), axis=0
            )
        if delete_projection_dict:
            for k in keys:
                del projections[k]
    print("Converting list to array..Completed")

    return projections_array


if __name__ == "__main__":
    test_dict = {"option_1": "a", "option_2": "b", "option_3": "c"}
    result = generate_input_user_prompt(
        load_object_type_string="experiment",
        options_list=test_dict.keys(),
        options_info_list=test_dict.values(),
        options_info_type_string="test_type_string",
    )
    print("returned: ", result)
