from typing import Union
import h5py
import numpy as np
import dataclasses
from enum import StrEnum
from numbers import Number
from llama.data_structures.projections import Projections, ShiftManager
from llama.task import LaminographyAlignmentTask


def save_task(task: LaminographyAlignmentTask, file_path: str, exclude: list[str] = []):
    save_attr_strings = ["complex_projections", "phase_projections"]
    with h5py.File(file_path, "w") as h5_obj:
        for attr in save_attr_strings:
            if (
                attr in task.__dict__.keys()
                and getattr(task, attr) is not None
                and attr not in exclude
            ):
                save_projections(getattr(task, attr), file_path, attr, h5_obj)
        save_options(task.options, h5_obj.create_group("options"))


def save_projections(projections: Projections, file_path: str, group_name: str, h5_obj: h5py.File):
    if projections.probe_positions is not None:
        positions = projections.probe_positions.data
    else:
        positions = None
    save_attr_dict = {
        "data": projections.data,
        "angles": projections.angles,
        "masks": projections.masks,
        "center_of_rotation": projections.center_of_rotation,
        "positions": positions,
        "pixel_size": projections.pixel_size,
    }
    h5_group = h5_obj.create_group(group_name)
    # Save all elements from save_attr_dict to the .h5 file
    save_generic_data_structure_to_h5(save_attr_dict, h5_group)
    # Save projection options
    save_options(projections.options, h5_group.create_group("options"))
    print(f"Array saved to {file_path}")


def save_options(obj, h5_obj: Union[h5py.Group, h5py.File]):
    for field_name, value in obj.__dict__.items():
        if dataclasses.is_dataclass(value):
            # Recursively handle dataclasses
            save_options(value, h5_obj.create_group(field_name))
        elif isinstance(value, StrEnum):
            # Handle enums
            h5_obj.create_dataset(field_name, data=value._value_)
        elif value is not None:
            h5_obj.create_dataset(field_name, data=value)
        else:
            pass


def save_generic_data_structure_to_h5(d: dict, h5_obj: Union[h5py.Group, h5py.File]):
    "Create h5 datasets for all items in the passed in dict"
    # This will need to be updated any time you are adding variables whose type
    # doesn't correspond to any of the if/else statements
    for value_name, value in d.items():
        if value is None:
            return
        elif dataclasses.is_dataclass(value):
            # Recursively handle dataclasses
            save_generic_data_structure_to_h5(value, h5_obj.create_group(value_name))
        elif check_if_standard_dataset(value):
            h5_obj.create_dataset(value_name, data=value)
        elif check_if_list_of_numpy_arrays(value):
            sub_group = h5_obj.create_group(value_name)
            for i, list_entry in enumerate(value):
                sub_group.create_dataset(str(i), data=list_entry)
        else:
            raise TypeError(
                f"{value_name} has unaccounted for type {type(value)}!\n"
                + "Update this function in order to properly save it."
            )


def check_if_standard_dataset(value):
    return isinstance(value, Union[np.ndarray, Number]) or (
        isinstance(value, list) and isinstance(value[0], Number)
    )


def check_if_list_of_numpy_arrays(value):
    return isinstance(value, list) and isinstance(value[0], np.ndarray)


def save_shift_manager(shift_manager: ShiftManager):
    pass
