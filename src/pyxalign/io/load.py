import h5py
from dataclasses import fields, is_dataclass
import enum

from pyxalign.data_structures.projections import (
    Projections,
    ComplexProjections,
    PhaseProjections,
    Projections,
    ShiftManager,
    TransformTracker,
)
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.api.options.task import AlignmentTaskOptions
from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.enums import SpecialValuePlaceholder, ShiftType
from pyxalign.api.types import c_type, r_type

from typing import Type, TypeVar, Union

OptionsClass = TypeVar("OptionsClass")


def load_task(file_path: str, exclude: list[str] = []) -> LaminographyAlignmentTask:
    print("Loading task from", file_path, "...")

    with h5py.File(file_path, "r") as h5_obj:
        # Load projections
        loaded_projections = load_projections(h5_obj, exclude)

        # Insert projections into task along with saved task options
        task = LaminographyAlignmentTask(
            options=load_options(h5_obj["options"], AlignmentTaskOptions),
            complex_projections=loaded_projections["complex_projections"],
            phase_projections=loaded_projections["phase_projections"],
        )

        print("Loading complete")

    return task


def load_projections(
    h5_obj: Union[h5py.Group, h5py.File], exclude: list[str] = []
) -> dict[str, Union[Projections, None]]:
    projections_map = {
        "complex_projections": ComplexProjections,
        "phase_projections": PhaseProjections,
    }
    loaded_projections: dict[str, Projections] = {
        "complex_projections": None,
        "phase_projections": None,
    }
    for group, projection_class in projections_map.items():
        if group in h5_obj.keys() and group not in exclude:
            # Create TransformTracker object
            transform_tracker = TransformTracker(
                rotation=h5_obj[group]["rotation"][()],
                shear=h5_obj[group]["shear"][()],
                downsample=h5_obj[group]["downsample"][()],
            )

            # Create ShiftManager object
            shift_manager = ShiftManager(n_projections=len(h5_obj[group]["angles"][()]))
            shift_manager.past_shifts = load_list_of_arrays(h5_obj[group], "applied_shifts")
            if "staged_shift" in h5_obj[group].keys():
                staged_shift = h5_obj[group]["staged_shift"][()]
                if "staged_shift_function_type" in h5_obj[group].keys():
                    staged_shift_function_type =  h5_obj[group]["staged_shift_function_type"][()]
                    if is_null_type(staged_shift_function_type):
                        staged_shift_function_type = handle_null_type(staged_shift_function_type)
                    else:
                        staged_shift_function_type = staged_shift_function_type.decode()
                        staged_shift_function_type = ShiftType(staged_shift_function_type)
                else:
                    staged_shift_function_type = None
                shift_manager.stage_shift(
                    shift=staged_shift,
                    function_type=staged_shift_function_type,
                )

            # Create Projections object
            if "file_paths" in h5_obj[group].keys():
                file_paths = load_list_of_arrays(h5_obj[group], "file_paths")
            else:
                file_paths = None
            loaded_projections[group] = projection_class(
                projections=h5_obj[group]["data"][()],
                angles=h5_obj[group]["angles"][()],
                scan_numbers=h5_obj[group]["scan_numbers"][()],
                options=load_options(h5_obj[group]["options"], ProjectionOptions),
                center_of_rotation=h5_obj[group]["center_of_rotation"][()],
                masks=load_array(h5_obj[group], "masks"),
                probe=load_array(h5_obj[group], "probe"),
                probe_positions=load_list_of_arrays(h5_obj[group], "positions"),
                transform_tracker=transform_tracker,
                shift_manager=shift_manager,
                skip_pre_processing=True,
                add_center_offset_to_positions=False,
                file_paths=file_paths
            )

            if "dropped_scan_numbers" in h5_obj[group].keys():
                dropped_scan_numbers = h5_obj[group]["dropped_scan_numbers"][()]
                if is_null_type(dropped_scan_numbers):
                    dropped_scan_numbers = handle_null_type(dropped_scan_numbers)
                loaded_projections[group].dropped_scan_numbers = list(dropped_scan_numbers)
    return loaded_projections


def load_array(h5_obj: h5py.Group, name: str):
    if name in h5_obj.keys():
        value = h5_obj[name][()]
        if is_null_type(value):
            return handle_null_type(value)
        else:
            return value


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


def load_options(h5_obj: h5py.Group, options_class: Type[OptionsClass]) -> OptionsClass:
    return dict_to_dataclass(options_class=options_class, data=h5_to_dict(h5_obj))


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


def is_null_type(value):
    if isinstance(value, bytes):
        value = value.decode()
        return value.lower() in SpecialValuePlaceholder.__members__.values()
    else:
        return False


def handle_null_type(value):
    value = value.decode()
    if value == SpecialValuePlaceholder.NONE:
        return None
    elif value == SpecialValuePlaceholder.EMPTY_LIST:
        return []


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
