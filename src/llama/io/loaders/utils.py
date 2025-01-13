from typing import Optional, Union
import traceback
import h5py
import numpy as np


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
    experiment_string = f"{index+1}. {option_string} ("
    if options_info_list is not None:
        experiment_string += f"{options_info_list[index]}"
        if options_info_type_string is not None:
            experiment_string += f" {options_info_type_string}"
    return experiment_string + ")\n"

def is_valid_option_provided(use_option: str, options_list: list[str]) -> Union[tuple, None]:
    # Use pre-provided option if it was passed in
    if use_option is not None:
        try:
            index = list(options_list).index(use_option)
            return (index, use_option)
        except ValueError:
            print("Provided option is not allowed because it is not available in `options_list`")
            print(traceback.format_exc())

def prompt_input_processing(options_list: list[str], options_info_list: Optional[list[str]]):
    # Ensure inputs are lists
    options_list = list(options_list)
    if options_info_list is not None:
        options_info_list = list(options_info_list)
    return options_list, options_info_list

def generate_single_input_user_prompt(
    load_object_type_string: str,
    options_list: list[str],
    options_info_list: Optional[list[str]] = None,
    options_info_type_string: Optional[str] = None,
    use_option: Optional[str] = None,
) -> tuple[int, str]:
    provided_option = is_valid_option_provided(use_option, options_list)
    if provided_option is not None:
        return provided_option
    options_list, options_info_list = prompt_input_processing(options_list, options_info_list)
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


def generate_list_input_user_prompt(
    load_object_type_string: str,
    options_list: list[str],
    options_info_list: Optional[list[str]] = None,
    options_info_type_string: Optional[str] = None,
    use_option: Optional[str] = None,
) -> tuple[int, str]:
    provided_option = is_valid_option_provided(use_option, options_list)
    if provided_option is not None:
        return provided_option
    options_list, options_info_list = prompt_input_processing(options_list, options_info_list)
    # Generate the user prompt
    prompt = (f"Select the {load_object_type_string} to load\n" 
              + "Enter inputs as a series of integers seperated by spaces:\n")
    for index, option_string in enumerate(options_list):
        prompt += generate_experiment_description(
            option_string, index, options_info_list, options_info_type_string
        )
    # Prompt the user to make a selection
    allowed_inputs = range(0, len(options_list))
    while True:
        try:
            user_input = input(prompt)
            input_list = np.array(parse_space_delimited_integers(user_input)) - 1
            for input_index in input_list:
                if input_index not in allowed_inputs:
                    raise ValueError
            else:
                print(f"Selected options {input_list}")
                return (input_list, [options_list[input_index] for input_index in input_list])
        except ValueError:
            print(
                "Invalid input. Please enter a space delimited list whose elements are"
                + f"between 1 and {allowed_inputs[-1]}."
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
    # try:
    # Split the string by spaces
    parts = input_string.split()
    
    # Check if all parts are valid integers
    # if all(part.isdigit() or (part.startswith('-') and part[1:].isdigit()) for part in parts):
    return [int(part) for part in parts]
    # except Exception as e:
    #     return None


def get_boolean_user_input(prompt: str) -> bool:
    while True:
        user_input = input(f"{prompt} (y/n): ").strip().lower()
        if user_input in {"y", "yes"}:
            return True
        elif user_input in {"n", "no"}:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


if __name__ == "__main__":
    test_dict = {"option_1": "a", "option_2": "b", "option_3": "c"}
    result = generate_single_input_user_prompt(
        load_object_type_string="experiment",
        options_list=test_dict.keys(),
        options_info_list=test_dict.values(),
        options_info_type_string="test_type_string",
    )
    print("returned: ", result)
