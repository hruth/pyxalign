from typing import Sequence, Union
import h5py
import numpy as np
import dataclasses
from enum import StrEnum
from numbers import Number
from llama.api.enums import SpecialValuePlaceholder
import tifffile as tiff


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
    images = images.copy()
    if min is None:
        min = images.min()
    if max is None:
        max = images.max()
    delta = max - min
    images[images < min] = min
    images[images > max] = max
    return (65535 * (images - min) / delta).astype(np.uint16)


def save_array_as_tiff(
    images: np.ndarray,
    file_path: str,
    min: float = None,
    max: float = None,
    divide_into_smaller_files: bool = True,
):
    images_uint16 = convert_to_uint_16(images, min, max)
    if divide_into_smaller_files:
        # If the tiff file is too large, imagej will not be able to open it
        max_file_size = 4 * 1e9  # max file size that imagej will tolerate is 4 GB
        if images_uint16.nbytes > max_file_size:
            n_files = int(np.ceil(images_uint16.nbytes / max_file_size))
            n_layers = len(images_uint16)
            layers_per_file = int(np.ceil(n_layers / n_files))
            path, ext = os.path.splitext(file_path)
            for i in range(n_files):
                selected_layers = images_uint16[i * layers_per_file : (i + 1) * layers_per_file]
                selection_file_path = path + f"_{i+1}_of_{n_files}" + ext
                tiff.imwrite(selection_file_path, selected_layers)
        else:
            tiff.imwrite(file_path, images_uint16)
    print(f"File saved to: {file_path}")
