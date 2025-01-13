from abc import ABC
from typing import Optional, Self, TypeVar
import numpy as np
import traceback


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
    use_option: Optional[str] = None,
) -> tuple[int, str]:
    # Use pre-provided option if it was passed in
    if use_option is not None:
        try:
            index = list(options_list).index(use_option)
            return (index, use_option)
        except ValueError:
            print("Provided option is not allowed because it is not available in `options_list`")
            print(traceback.format_exc())
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
    result = generate_selection_user_prompt(
        load_object_type_string="experiment",
        options_list=test_dict.keys(),
        options_info_list=test_dict.values(),
        options_info_type_string="test_type_string",
    )
    print("returned: ", result)
