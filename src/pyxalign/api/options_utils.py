import dataclasses
from dataclasses import is_dataclass, fields
import numpy as np
import pyxalign.api.options as opts


def set_all_device_options(options, device_options: "opts.DeviceOptions"):
    "Set all device options in the input `options` dataclass to the valeus in `device_options`"

    for k, v in options.__dict__.items():
        # I am not using isinstance here because it does not always work as expectd when
        # using the reload_module_recursively and refresh_task function.
        if is_dataclass(v) and v.__class__.__qualname__ == opts.DeviceOptions.__qualname__:
            options.__dict__[k] = device_options
        elif is_dataclass(v):
            set_all_device_options(v, device_options)
        else:
            pass


def print_options(options, prepend="- "):
    for k, v in options.__dict__.items():
        if is_any_dataclass_instance(v):
            print(f"{prepend}{k} options")
            print_options(v, "     " + prepend)
        else:
            print(f"{prepend}{k}: {v}")


def is_any_dataclass_instance(obj):
    return is_dataclass(obj) and not isinstance(obj, type)


def get_all_attribute_names(obj, parent_prefix=None, level=0, max_level=999):
    """
    Recursively collect all attribute names of a nested dataclass object.

    If an attribute itself is another dataclass,
    the nested attributes will be prefixed with "<parent>."
    to reflect the nesting structure.
    """
    if parent_prefix is None:
        parent_prefix = ""

    if not is_dataclass(obj):
        raise TypeError("obj must be a dataclass instance.")

    paths = []
    for f in fields(obj):
        field_name = f.name
        # Build the dotted name if we have a parent prefix
        dotted_name = f"{parent_prefix}.{field_name}" if parent_prefix else field_name

        value = getattr(obj, field_name)
        # If the value is another dataclass instance, recurse
        if value is not None and is_dataclass(value) and level < max_level:
            # Add this field name itself, then all nested names
            paths.extend(get_all_attribute_names(value, dotted_name, level=level + 1))
        else:
            # It's a regular field (or None) => just add
            paths.append(dotted_name)
    return paths


def print_all_attributes(obj):
    """
    Print all attribute names of a (possibly nested) dataclass object.
    """
    names = get_all_attribute_names(obj)
    for name in names:
        print(name)
