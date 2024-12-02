from typing import Union, Optional
import h5py
import numpy as np
import dataclasses
from enum import StrEnum

from llama.projections import Projections, ShiftManager
from llama.task import LaminographyAlignmentTask
from llama.api.options.task import AlignmentTaskOptions


def save_task(task: LaminographyAlignmentTask, file_path: str, exclude: list[str] = []):
    save_attr_strings = ["complex_projections", "phase_projections"]
    with h5py.File(file_path, "w") as h5_file_obj:
        for attr in save_attr_strings:
            if attr in task.__dict__.keys() and attr not in exclude:
                save_projections(getattr(task, attr), file_path, attr, h5_file_obj)
        save_options(task.options, "options", h5_file_obj)


def save_projections(projections: Projections, file_path: str, group_name: str, h5_file_obj):
    save_attr_strings = ["data", "angles", "masks"]
    group = h5_file_obj.create_group(group_name)
    for attr in save_attr_strings:
        if attr in projections.__dict__.keys():
            group.create_dataset(attr, data=getattr(projections, attr))

    save_options(projections.options, group_name + "/options", h5_file_obj)
    print(f"Array saved to {file_path}")


def save_shift_manager(shift_manager: ShiftManager):
    pass


def save_options(
    obj,
    group_name,
    h5_file_obj: Union[h5py.Group, h5py.File],
    group: Optional[h5py.Group] = None,
):
    if group is None:
        group = h5_file_obj.create_group(group_name)
    for field_name, value in obj.__dict__.items():
        if dataclasses.is_dataclass(value):
            subgroup = group.create_group(field_name)
            save_options(value, field_name, h5_file_obj, subgroup)
        elif isinstance(value, StrEnum):
            group.attrs[field_name] = value._value_
        elif isinstance(value, (int, float, str, bool)):
            group.attrs[field_name] = value
        elif isinstance(value, (list, tuple)):
            # Handle lists
            group.create_dataset(field_name, data=value)


if __name__ == "__main__":
    options = AlignmentTaskOptions()
    with h5py.File("test_save_options.h5", "w") as h5_file_obj:
        save_options(options, "options", h5_file_obj)
