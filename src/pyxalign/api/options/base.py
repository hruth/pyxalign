from typing import Union, get_origin, get_args
import dataclasses
from dataclasses import fields
import logging
import enum
import yaml

from numpy import ndarray
import numpy as np


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BaseOptions:
    def __setattr__(self, name, value):
        # # Check if the attribute already exists in the class fields.
        # if name not in {f.name for f in dataclasses.fields(self)}:
        #     raise AttributeError(f"{name} is not a valid field in {self.__class__.__name__}.")
        # # If it exists, allow setting the value.
        # super().__setattr__(name, value)

        # Check if the attribute already exists in the class fields.
        if name in {f.name for f in dataclasses.fields(self)}:
            # If it exists, allow setting the value.
            super().__setattr__(name, value)
        # backwards compatibility elifs
        elif name == "phase_unwrap_masks":
            super().__setattr__("phase_unwrap_masks_from_position", value)
        elif name == "projection_matching_masks":
            super().__setattr__("projection_matching_masks_from_position", value)
        else:
            raise AttributeError(f"{name} is not a valid field in {self.__class__.__name__}.")

    def check(self, *args, **kwargs) -> None:
        """Check if options values are valid."""
        return

    def resolve_type(self, ann_type) -> type:
        """Resolve annotation to underlying type (handles Optional, etc.)."""
        origin = get_origin(ann_type)
        if origin is Union:
            args = get_args(ann_type)
            # Drop NoneType from Optional[...]
            return next((arg for arg in args if arg is not type(None)), None)
        return ann_type

    def get_non_data_fields(self) -> dict:
        """Get fields that do not contain large arrays or tensors."""
        d = self.__dict__.copy()
        return d

    def get_dict(self) -> dict:
        """Get a dictionary representation of the options."""
        d = self.get_non_data_fields()
        for k, v in d.items():
            if isinstance(v, BaseOptions) or (v.__class__.__bases__[0] == "pyxalign.api.options.base.BaseOptions"): # doesnt work when reloading/debugging
                d[k] = v.get_dict()
            else:
                d[k] = jsonize(v)
        return d

    def load_from_dict(self, d: dict) -> "BaseOptions":
        """Load options from a dictionary."""
        for k, v in d.items():
            if not hasattr(self, k):
                continue
            field_type = self.resolve_type(self.get_field_type(k))
            if isinstance(field_type, type) and issubclass(field_type, BaseOptions):
                self.__setattr__(k, self.resolve_type(self.get_field_type(k))().load_from_dict(v))
            elif (
                isinstance(field_type, type)
                and issubclass(field_type, enum.StrEnum)
                and isinstance(v, str)
            ):
                self.__setattr__(k, field_type(v))
                # self.__setattr__(k, field_type[v.upper()])
            elif get_origin(field_type) is tuple and isinstance(v, list):
                # Convert lists back to tuples when the field type is a tuple
                self.__setattr__(k, tuple(v))
            else:
                self.__setattr__(k, v)
        return self

    def get_field_type(self, name: str) -> type:
        """Get the type of a field."""
        for f in fields(self):
            if f.name == name:
                return f.type
        raise ValueError(f"Field {name} not found in {self.__class__.__name__}.")

    def save_to_dict(self, path: str):
        with open(path, "w") as f:
            yaml.safe_dump(self.get_dict(), f, default_flow_style=False, sort_keys=False)

    def load_from_path(self, path:str)-> "BaseOptions":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return self.load_from_dict(d)



def jsonize(val):
    """Convert a value to a JSON-serializable object."""
    if isinstance(val, np.generic):
        return val.item()
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, enum.Enum):
        return val.value
    elif isinstance(val, tuple):
        return list(val)
    elif isinstance(val, list):
        if isinstance(val[0], np.int64):
            val = [int(x) for x in val]
        else:
            return val
    elif isinstance(val, (list, dict, str, int, float, bool, type(None))):
        return val
    else:
        print(type(val))
        raise TypeError(f"Object of type {type(val).__name__} is not JSON serializable")
