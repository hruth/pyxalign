from dataclasses import fields, is_dataclass
import enum
from typing import Type, Union, TypeVar
import h5py

from pyxalign.api.enums import SpecialValuePlaceholder

OptionsClass = TypeVar("OptionsClass")


def handle_null_type(value):
    value = value.decode()
    if value == SpecialValuePlaceholder.NONE:
        return None
    elif value == SpecialValuePlaceholder.EMPTY_LIST:
        return []


def is_null_type(value):
    if isinstance(value, bytes):
        value = value.decode()
        return value.lower() in SpecialValuePlaceholder.__members__.values()
    else:
        return False


def load_list_of_arrays(h5_obj: h5py.Group, name: str):
    if name in h5_obj.keys():
        if isinstance(h5_obj[name], h5py.Dataset):
            value = h5_obj[name][()]
            if is_null_type(value):
                # pass
                return handle_null_type(value)
        else:
            n_arrays = len(h5_obj[name])
            list_of_arrays = list(range(n_arrays))
            for i in range(n_arrays):
                entry = h5_obj[name][str(i)][()]
                if isinstance(entry, bytes):
                    entry = entry.decode()
                list_of_arrays[i] = entry
                # list_of_arrays[i] = h5_obj[name][str(i)][()]
            return list_of_arrays


def h5_to_dict(h5_obj: Union[h5py.Group, h5py.File]):
    """
    Recursively converts an HDF5 group or file into a Python dictionary.

    Args:
        h5_obj: h5py File or Group object.

    Returns:
        dict: A dictionary representation of the HDF5 structure.
    """
    result = {}

    # Add datasets and attributes
    for key, item in h5_obj.items():
        if isinstance(item, h5py.Group):
            # Recurse into groups
            result[key] = h5_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            # Convert datasets to their value
            result[key] = item[()]  # Use [()] to retrieve the data
            if isinstance(result[key], bytes):
                if is_null_type(result[key]):
                    result[key] = handle_null_type(result[key])
                else:
                    result[key] = result[key].decode()

    # Add attributes
    for attr in h5_obj.attrs:
        result[attr] = h5_obj.attrs[attr]

    return result


def dict_to_dataclass(
    options_class: Type[OptionsClass], data: dict, options_path_string: str = ""
) -> OptionsClass:
    """
    Recursively converts a dictionary into an options dataclass instance.

    In the options dataclasses, there typically are one or more attributes that are
    other options classes. If the names of these attributes are changed (don't confuse
    this with the class name) then this function will not loaded those values in,
    because the options are saved in a dataset named after the attribute.

    If the attributes names are ever changed and you still need this to load those
    attributes, you can make a function that maps the all of the past attribute names
    to the current attribute name and update this function to use that.
    """
    if not is_dataclass(options_class):
        raise ValueError(f"{options_class} must be a dataclass")
    arrow_symbol = " \u2192 "
    if options_path_string == "":
        options_path_string = options_class.__qualname__ + arrow_symbol

    field_values = {}
    for field in fields(options_class):
        field_name = field.name
        field_type = field.type
        if field_name not in data:
            print("WARNING: expected value not found")
            print(
                f"'{field_name}' not found in saved options, using default values for '{field_name}'."
            )
            print(options_path_string + field_name + "\n")
            continue
        value = data[field_name]

        # Check if the field is a dataclass
        if is_dataclass(field_type):
            field_values[field_name] = dict_to_dataclass(
                field_type, value, options_path_string + field_name + "."
            )
        else:
            if isinstance(value, str):
                # Convert to str enum if necessary
                expected_type = options_class.__annotations__[field_name]
                if isinstance(expected_type, enum.EnumType):
                    value = expected_type[value.upper()]
            field_values[field_name] = value

    return options_class(**field_values)


def load_options(h5_obj: h5py.Group, options_class: Type[OptionsClass]) -> OptionsClass:
    return dict_to_dataclass(options_class=options_class, data=h5_to_dict(h5_obj))


def load_array(h5_obj: h5py.Group, name: str):
    if name in h5_obj.keys():
        value = h5_obj[name][()]
        if is_null_type(value):
            return handle_null_type(value)
        else:
            return value
