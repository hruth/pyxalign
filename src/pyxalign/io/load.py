from numbers import Complex
import h5py

from pyxalign.api.enums import ProjectionType
from pyxalign.data_structures import xrf_projections
from pyxalign.data_structures.projections import (
    Projections,
    ComplexProjections,
    PhaseProjections,
    Projections,
    ShiftManager,
    TransformTracker,
)
from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.enums import ShiftType
from pyxalign.api.types import c_type, r_type

from typing import Optional, TypeVar, Union

from pyxalign.io.utils import (
    handle_null_type,
    is_null_type,
    load_array,
    load_list_of_arrays,
    load_options,
)


def load_ptycho_projections(
    task_h5_obj: Union[h5py.Group, h5py.File], exclude: list[ProjectionType] = []
) -> dict[str, Union[Projections, None]]:
    projections_map = {
        "complex_projections": ProjectionType.COMPLEX,
        "phase_projections": ProjectionType.PHASE,
    }
    loaded_projections: dict[str, Projections] = {
        "complex_projections": None,
        "phase_projections": None,
    }
    for group, projection_type in projections_map.items():
        if group in task_h5_obj.keys() and group not in exclude:
            loaded_projections[group] = load_projections_object(task_h5_obj[group], projection_type)
    return loaded_projections


def load_xrf_projections(
    task_h5_obj: Union[h5py.Group, h5py.File], exclude_channels: Optional[list[str]] = None
) -> dict[str, PhaseProjections]:
    if exclude_channels is None:
        exclude_channels = []
    xrf_projections_dict = {}
    all_channels = list(task_h5_obj["projections"].keys())
    for channel in all_channels:
        if channel not in exclude_channels:
            projection_channel_h5_group = task_h5_obj["projections"][channel]
            xrf_projections_dict[channel] = load_projections_object(
                proj_h5_obj=projection_channel_h5_group, projection_type=ProjectionType.PHASE
            )
    return xrf_projections_dict

def load_projections_object(
    proj_h5_obj: Union[h5py.Group, h5py.File], projection_type: ProjectionType
):
    # select the right class
    if projection_type == ProjectionType.COMPLEX:
        projection_class = ComplexProjections
    elif projection_type == ProjectionType.PHASE:
        projection_class = PhaseProjections

    # Create TransformTracker object
    transform_tracker = TransformTracker(
        rotation=proj_h5_obj["rotation"][()],
        shear=proj_h5_obj["shear"][()],
        downsample=proj_h5_obj["downsample"][()],
    )

    # Create ShiftManager object
    shift_manager = ShiftManager(n_projections=len(proj_h5_obj["angles"][()]))
    shift_manager.past_shifts = load_list_of_arrays(proj_h5_obj, "applied_shifts")
    if "staged_shift" in proj_h5_obj.keys():
        staged_shift = proj_h5_obj["staged_shift"][()]
        if "staged_shift_function_type" in proj_h5_obj.keys():
            staged_shift_function_type = proj_h5_obj["staged_shift_function_type"][()]
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
    if "file_paths" in proj_h5_obj.keys():
        file_paths = load_list_of_arrays(proj_h5_obj, "file_paths")
    else:
        file_paths = None
    projections = projection_class(
        projections=proj_h5_obj["data"][()],
        angles=proj_h5_obj["angles"][()],
        scan_numbers=proj_h5_obj["scan_numbers"][()],
        options=load_options(proj_h5_obj["options"], ProjectionOptions),
        center_of_rotation=proj_h5_obj["center_of_rotation"][()],
        masks=load_array(proj_h5_obj, "masks"),
        probe=load_array(proj_h5_obj, "probe"),
        probe_positions=load_list_of_arrays(proj_h5_obj, "positions"),
        transform_tracker=transform_tracker,
        shift_manager=shift_manager,
        skip_pre_processing=True,
        add_center_offset_to_positions=False,
        file_paths=file_paths,
    )

    if "dropped_scan_numbers" in proj_h5_obj.keys():
        dropped_scan_numbers = proj_h5_obj["dropped_scan_numbers"][()]
        if is_null_type(dropped_scan_numbers):
            dropped_scan_numbers = handle_null_type(dropped_scan_numbers)
        projections.dropped_scan_numbers = list(dropped_scan_numbers)

    return projections
