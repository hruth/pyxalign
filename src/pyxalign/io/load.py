import h5py

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
from pyxalign.api.enums import ShiftType
from pyxalign.api.types import c_type, r_type

from typing import TypeVar, Union

from pyxalign.io.utils import (
    handle_null_type,
    is_null_type,
    load_array,
    load_list_of_arrays,
    load_options,
)


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
                    staged_shift_function_type = h5_obj[group]["staged_shift_function_type"][()]
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
                file_paths=file_paths,
            )

            if "dropped_scan_numbers" in h5_obj[group].keys():
                dropped_scan_numbers = h5_obj[group]["dropped_scan_numbers"][()]
                if is_null_type(dropped_scan_numbers):
                    dropped_scan_numbers = handle_null_type(dropped_scan_numbers)
                loaded_projections[group].dropped_scan_numbers = list(dropped_scan_numbers)
    return loaded_projections
