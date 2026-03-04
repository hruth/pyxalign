import yaml
import copy
from pyxalign.api.options.alignment import ProjectionMatchingOptions
from pyxalign.api.options.base import BaseOptions


def get_updated_options(default_options: BaseOptions, new_options_dict: dict) -> BaseOptions:
    updated_options = copy.deepcopy(default_options)
    return _update_options_recursively(updated_options, new_options_dict)


def _update_options_recursively(options: BaseOptions, new_options_dict: dict):
    field_names = new_options_dict.keys()
    for field_name in field_names:
        field_value = getattr(options, field_name)
        if isinstance(field_value, BaseOptions):
            _update_options_recursively(field_value, new_options_dict[field_name])
        else:
            setattr(options, field_name, new_options_dict[field_name])
    return options


def get_projection_matching_sequence_options(
    default_options: ProjectionMatchingOptions, updated_settings_dicts: dict,#file_path: str
) -> list[ProjectionMatchingOptions]:
    """
    - file path is the path to the autorunner options path
    """
    # with open(file_path, "r") as f:
    #     autorunner_settings = yaml.safe_load(f)
    # updated_settings_dicts = autorunner_settings["ProjectionMatching"]["Sequence"]
    pma_options_sequence = []
    for d in updated_settings_dicts:
        pma_options_sequence += [get_updated_options(default_options, d)]
    return pma_options_sequence


def load_options_from_yaml(file_path: str, options: BaseOptions):
    with open(file_path, "r") as f:
        options_dict = yaml.safe_load(f)
    options.load_from_dict(options_dict)
    return options


def get_autorunner_options_dict(file_path: str):
    with open(file_path, "r") as f:
        options_dict = yaml.safe_load(f)
    return options_dict