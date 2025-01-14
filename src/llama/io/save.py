from typing import Union, Optional
import h5py
import numpy as np
import dataclasses
from enum import StrEnum

from llama.projections import Projections, ShiftManager
from llama.task import LaminographyAlignmentTask
from llama.api.options.task import AlignmentTaskOptions


def save_task(task: LaminographyAlignmentTask, file_path: str, exclude: list[str] = []):
    save_attr_strings = ["complex_projections", "phase_projections", "laminogram"]
    with h5py.File(file_path, "w") as h5_obj:
        for attr in save_attr_strings:
            if (
                attr in task.__dict__.keys()
                and getattr(task, attr) is not None
                and attr not in exclude
            ):
                save_projections(getattr(task, attr), file_path, attr, h5_obj)
        save_options(task.options, h5_obj.create_group("options"))


def save_projections(
    projections: Projections, file_path: str, group_name: str, h5_obj: h5py.File
):
    save_attr_strings = ["data", "angles", "masks", "center_of_rotation"]
    h5_group = h5_obj.create_group(group_name)
    for attr in save_attr_strings:
        if attr in projections.__dict__.keys():
            if hasattr(projections, attr) and getattr(projections, attr) is not None:
                h5_group.create_dataset(attr, data=getattr(projections, attr))

    save_options(projections.options, h5_group.create_group("options"))
    print(f"Array saved to {file_path}")


def save_shift_manager(shift_manager: ShiftManager):
    pass


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
