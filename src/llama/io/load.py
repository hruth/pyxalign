import h5py
import numpy as np
from dataclasses import fields, is_dataclass

from llama.projections import ComplexProjections, PhaseProjections
from llama.task import LaminographyAlignmentTask
from llama.api.options.task import AlignmentTaskOptions
from llama.api.options.projections import ProjectionOptions

from typing import Type, TypeVar, Union

T = TypeVar("T")


def load_task(file_path: str, exclude: list[str] = []) -> LaminographyAlignmentTask:
    def get_masks(group, h5_obj):
        if "masks" in h5_obj[group].keys():
            return h5_obj[group]["masks"][:]
        else:
            return None
    with h5py.File(file_path, "r") as h5_obj:
        group = "complex_projections"
        if group in h5_obj.keys() and group not in exclude:
            complex_projections = ComplexProjections(
                projections=h5_obj[group + "/data"][:],
                angles=h5_obj[group + "/angles"][:],
                options=load_options(h5_obj[group], ProjectionOptions),
                masks=get_masks(group, h5_obj),
                shift_manager=None,  # needs to be updated later
            )
        else:
            complex_projections = None

        group = "phase_projections"
        if group in h5_obj.keys() and group not in exclude:
            phase_projections = PhaseProjections(
                projections=h5_obj[group + "/data"][:],
                angles=h5_obj[group + "/angles"][:],
                options=load_options(h5_obj[group]["options"], ProjectionOptions),
                masks=get_masks(group, h5_obj),
                shift_manager=None,  # needs to be updated later
            )
        else:
            phase_projections = None

        task = LaminographyAlignmentTask(
            options=load_options(h5_obj["options"], AlignmentTaskOptions),
            complex_projections=complex_projections,
            phase_projections=phase_projections,
        )

    return task


def load_options(h5_obj, options_class: Type[T]) -> T:
    return dict_to_dataclass(cls=options_class, data=h5_to_dict(h5_obj))


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
                result[key] = result[key].decode()

    # Add attributes
    for attr in h5_obj.attrs:
        result[attr] = h5_obj.attrs[attr]

    return result


def dict_to_dataclass(cls: Type[T], data: dict) -> T:
    """
    Recursively converts a dictionary into a dataclass instance.
    """
    if not is_dataclass(cls):
        raise ValueError(f"{cls} must be a dataclass")

    field_values = {}
    for field in fields(cls):
        field_name = field.name
        field_type = field.type
        if field_name not in data:
            continue
        value = data[field_name]

        # Check if the field is a dataclass
        if is_dataclass(field_type):
            field_values[field_name] = dict_to_dataclass(field_type, value)
        else:
            field_values[field_name] = value

    return cls(**field_values)
