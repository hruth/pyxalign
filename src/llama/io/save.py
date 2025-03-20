from typing import Sequence, Union
import h5py
import numpy as np
import dataclasses
from enum import StrEnum
from numbers import Number
from llama.api.enums import SpecialValuePlaceholder
from llama.data_structures.projections import Projections, ShiftManager
from llama.task import LaminographyAlignmentTask
import tifffile as tiff


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
        save_generic_data_structure_to_h5(task.options, h5_obj.create_group("options"))


def save_projections(projections: Projections, file_path: str, group_name: str, h5_obj: h5py.File):
    if projections.probe_positions is not None:
        positions = projections.probe_positions.data
    else:
        positions = None
    save_attr_dict = {
        "data": projections.data,
        "angles": projections.angles,
        "scan_numbers": projections.scan_numbers,
        "masks": projections.masks,
        "center_of_rotation": projections.center_of_rotation,
        "positions": positions,
        "probe": projections.probe,
        "pixel_size": projections.pixel_size,
        "rotation": projections.transform_tracker.rotation,
        "shear": projections.transform_tracker.shear,
        "downsample": projections.transform_tracker.scale,
        "applied_shifts": projections.shift_manager.past_shifts,
        "staged_shift": projections.shift_manager.staged_shift,
    }
    h5_group = h5_obj.create_group(group_name)
    # Save all elements from save_attr_dict to the .h5 file
    save_generic_data_structure_to_h5(save_attr_dict, h5_group)
    # Save projection options
    save_generic_data_structure_to_h5(projections.options, h5_group.create_group("options"))
    print(f"Array saved to {file_path}")


def save_generic_data_structure_to_h5(d: dict, h5_obj: Union[h5py.Group, h5py.File]):
    "Create h5 datasets for all items in the passed in dict"
    # This will need to be updated any time you are adding variables whose type
    # doesn't correspond to any of the if/else statements
    if not isinstance(d, dict):
        d = d.__dict__
    for value_name, value in d.items():
        if dataclasses.is_dataclass(value):
            # Recursively handle dataclasses
            save_generic_data_structure_to_h5(value, h5_obj.create_group(value_name))

        elif isinstance(value, list) and len(value) == 0:
            # Empty lists
            h5_obj.create_dataset(value_name, data=SpecialValuePlaceholder.EMPTY_LIST._value_)

        elif value is None:
            # None types
            h5_obj.create_dataset(value_name, data=SpecialValuePlaceholder.NONE._value_)

        elif isinstance(value, Union[bool, np.bool_]):
            h5_obj.create_dataset(value_name, data=value)

        elif isinstance(value, np.ndarray):
            # Arrays
            h5_obj.create_dataset(value_name, data=value, dtype=value.dtype)

        elif isinstance(value, Number):
            # Individual numbers
            h5_obj.create_dataset(value_name, data=value, dtype=type(value))

        elif isinstance(value, Sequence) and isinstance(value[0], Number):
            # Sequence (i.e. list, tuple) of numbers
            h5_obj.create_dataset(value_name, data=value, dtype=type(value[0]))

        elif isinstance(value, str):
            # Strings and string enums
            save_string_to_h5(h5_obj, value, value_name)

        elif isinstance(value, list) and (len(value) > 0 and isinstance(value[0], str)):
            # List of string enums
            sub_group = h5_obj.create_group(value_name)
            for i, list_entry in enumerate(value):
                save_string_to_h5(sub_group, list_entry[i], str(i))

        elif isinstance(value, list) and (len(value) > 0 and isinstance(value[0], np.ndarray)):
            # List of numpy arrays
            sub_group = h5_obj.create_group(value_name)
            for i, list_entry in enumerate(value):
                sub_group.create_dataset(str(i), data=list_entry, dtype=list_entry.dtype)

        else:
            print(f"WARNING: {value_name} not saved")


def save_string_to_h5(h5_obj: Union[h5py.Group, h5py.File], string: str, value_name: str):
    if isinstance(string, StrEnum):
        h5_obj.create_dataset(value_name, data=string._value_)
    else:
        h5_obj.create_dataset(value_name, data=string)


def save_options_to_h5_file(file_path: str, options):
    F = h5py.File(file_path, "w")
    save_generic_data_structure_to_h5(options, F)
    F.close()


def convert_to_uint_16(images: np.ndarray, min: float = None, max: float = None):
    if min is None:
        min = images.min()
    if max is None:
        max = images.max()
    delta = max - min
    images[images < min] = min
    images[images > max] = max
    return (65535 * (images - min) / delta).astype(np.uint16)


def save_array_as_tiff(images: np.ndarray, file_path: str, min: float = None, max: float = None):
    tiff.imwrite(file_path, convert_to_uint_16(images, min, max))