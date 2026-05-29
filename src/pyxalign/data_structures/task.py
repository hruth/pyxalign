from typing import Optional, Union
import numpy as np
import h5py
import copy

from pyxalign import gpu_utils
from pyxalign.api.options.alignment import ProjectionMatchingOptions
from pyxalign.data_structures.projections import (
    ComplexProjections,
    PhaseProjections,
    Projections,
    get_kwargs_for_copying_to_new_projections_object,
)
from pyxalign.alignment.cross_correlation import CrossCorrelationAligner
from pyxalign.alignment.pma_tracking import (
    PMASequence,
    PMASnapshot,
    crop_volume_for_recording,
)
from pyxalign.alignment.projection_matching import ProjectionMatchingAligner
from pyxalign.api.options.task import AlignmentTaskOptions
from pyxalign.api import enums
from pyxalign.api.types import r_type
from pyxalign.io.load import load_ptycho_projections
from pyxalign.io.save import save_generic_data_structure_to_h5
from pyxalign.io.utils import load_options_from_h5_group
from pyxalign.interactions.viewers.projection_matching import ProjectionMatchingViewer
from pyxalign.timing.timer_utils import clear_timer_globals

# from pyxalign.interactions.viewers.task import TaskViewer # causes circular imports
# import pyxalign.interactions.viewers.task as task_viewer
# import pyxalign.interactions.pma_runner as pma_runner


class LaminographyAlignmentTask:
    def __init__(
        self,
        options: Optional[AlignmentTaskOptions] = None,
        complex_projections: Optional[ComplexProjections] = None,
        phase_projections: Optional[PhaseProjections] = None,
    ):
        if options is None:
            options = AlignmentTaskOptions()
        self.options = options
        if phase_projections is None and complex_projections is None:
            raise Exception(
                "Projections must be included when creating an instance of LaminographyAlignmentTask"
            )

        self.complex_projections = complex_projections
        self.phase_projections = phase_projections
        self.pma_object: ProjectionMatchingAligner = None
        self.pma_gui_list: list[ProjectionMatchingViewer] = []
        self.pma_sequence = PMASequence()

    def get_cross_correlation_shift(
        self,
        projection_type: enums.ProjectionType = enums.ProjectionType.COMPLEX,
        illum_sum: np.ndarray = None,
        plot_results: bool = True,
    ) -> np.ndarray:
        clear_timer_globals()
        # Only for complex projections for now
        # Does this really need to be saved as an attribute?
        if projection_type == enums.ProjectionType.COMPLEX:
            projections = self.complex_projections
        else:
            projections = self.phase_projections
        self.cross_correlation_aligner = CrossCorrelationAligner(
            projections, self.options.cross_correlation
        )
        # Placeholder for actual illum_sum
        if illum_sum is None:
            self.illum_sum = np.ones_like(projections.data[0], dtype=r_type)
        else:
            self.illum_sum = illum_sum
        shift = self.cross_correlation_aligner.run(self.illum_sum)
        projections.shift_manager.stage_shift(
            shift=shift,
            function_type=enums.ShiftType.CIRC,
            alignment_options=self.options.cross_correlation,
            eliminate_wrapping=True,
        )
        if plot_results:
            projections.plot_shift(
                shift_type=enums.ShiftManagerMemberType.STAGED_SHIFT,
                title="Cross-correlation Shift",
            )
        print("Cross-correlation shift stored in shift_manager")
        return shift

    def get_projection_matching_shift(
        self,
        initial_shift: Optional[Union[np.ndarray, PMASnapshot]] = None,
        options: Optional[ProjectionMatchingOptions] = None,
    ) -> np.ndarray:
        # clear existing astra objects
        if self.pma_object is not None:
            if hasattr(self.pma_object, "aligned_projections"):
                self.pma_object.aligned_projections.volume.clear_astra_objects()

        # reset timers
        clear_timer_globals()

        # close old gui windows
        if self.options.projection_matching.interactive_viewer.close_old_windows:
            self.clear_pma_gui_list()
            if self.pma_object is not None and self.pma_object.gui is not None:
                self.pma_object.gui.close()
        else:
            self.pma_gui_list += [self.pma_object.gui]

        # assign options
        if options is None:
            options = self.options.projection_matching

        # If a prior snapshot is passed in, use its final shift as the
        # starting point for this run and record the parent link so the
        # alignment chain can be reconstructed later.
        parent_index: Optional[int] = None
        if isinstance(initial_shift, PMASnapshot):
            parent_snapshot = initial_shift
            for i, s in enumerate(self.pma_sequence.snapshots):
                if s is parent_snapshot:
                    parent_index = i
                    break
            initial_shift = parent_snapshot.compute_shift_relative_to(
                self.phase_projections
            )

        # snapshot the inputs to this PMA call before running
        self.pma_sequence.append(
            PMASnapshot.from_phase_projections(
                self.phase_projections,
                pma_options=options,
                initial_shift=initial_shift,
                parent_index=parent_index,
            )
        )

        # run the pma algorithm
        shift = self.run_projection_matching(
            self.phase_projections, initial_shift, options
        )
        last_snapshot = self.pma_sequence.snapshots[-1]
        last_snapshot.final_shift = np.asarray(shift).copy()

        # optionally record the post-PMA volume into the snapshot
        if options.pma_sequence.record_volume:
            try:
                volume_data = self.pma_object.aligned_projections.volume.data
            except AttributeError:
                volume_data = None
            if volume_data is not None:
                last_snapshot.volume = crop_volume_for_recording(
                    volume_data, options.pma_sequence
                )

        # Store the result in the ShiftManager object
        self.phase_projections.shift_manager.stage_shift(
            shift=shift,
            function_type=enums.ShiftType.FFT,
            alignment_options=self.options.projection_matching,
        )
        print("Projection-matching shift stored in shift_manager")

        return shift

    def clear_pma_gui_list(self):
        for gui in self.pma_gui_list:
            gui.close()
        self.pma_gui_list = []

    def get_unwrapped_phase(
        self, pinned_results: Optional[np.ndarray] = None, skip_pinning: bool = False
    ):
        if self.complex_projections is None:
            raise ValueError("No complex projections available for phase unwrapping")

        if not skip_pinning and pinned_results is None:
            if (
                self.phase_projections is not None
                and self.phase_projections.data.shape == self.complex_projections.data.shape
            ):
                pinned_results = gpu_utils.pin_memory(self.phase_projections.data)
            else:
                pinned_results = gpu_utils.create_empty_pinned_array(
                    self.complex_projections.data.shape, dtype=r_type
                )

        unwrapped_projections = self.complex_projections.unwrap_phase(pinned_results)
        kwargs = get_kwargs_for_copying_to_new_projections_object(
            self.complex_projections, include_projections_copy=False
        )
        self.phase_projections = PhaseProjections(projections=unwrapped_projections, **kwargs)
        # update with dropped scan numbers, angles, and file paths from complex projections
        self.phase_projections.dropped_scan_numbers = copy.copy(self.complex_projections.dropped_scan_numbers)
        self.phase_projections.dropped_angles = copy.copy(self.complex_projections.dropped_angles)
        self.phase_projections.dropped_file_paths = copy.copy(self.complex_projections.dropped_file_paths)

    def save_task(
        self,
        file_path: str,
        exclude: list[str] = [],
        save_pma_sequence_volumes: bool = False,
    ):
        save_attr_strings = ["complex_projections", "phase_projections"]
        with h5py.File(file_path, "w") as h5_obj:
            for attr in save_attr_strings:
                if (
                    attr in self.__dict__.keys()
                    and getattr(self, attr) is not None
                    and attr not in exclude
                ):
                    # save_projections(getattr(self, attr), file_path, attr, h5_obj)
                    projection: Projections = getattr(self, attr)
                    projection._save_projections_object(h5_obj=h5_obj.create_group(attr))
            save_generic_data_structure_to_h5(self.options, h5_obj.create_group("options"))
            # Persist the PMA sequence alongside the task only when it
            # actually has snapshots; old task files have no such group
            # and the loader treats its absence as "empty sequence".
            if len(self.pma_sequence) > 0:
                self.pma_sequence._save_to_group(
                    h5_obj.create_group("pma_sequence"),
                    include_volumes=save_pma_sequence_volumes,
                )
            print(f"task saved to {h5_obj.file.filename}{h5_obj.name}")

    def run_projection_matching(
        self,
        phase_projections: PhaseProjections,
        initial_shift: np.ndarray,
        projection_matching_options: ProjectionMatchingOptions,
    ) -> tuple[ProjectionMatchingAligner, np.ndarray]:
        # Initialize the projection-matching alignment object
        self.pma_object = ProjectionMatchingAligner(phase_projections, projection_matching_options)
        try:
            if self.pma_object.options.interactive_viewer.update.enabled:
                # Run PMA algorithm
                shift = self.pma_object.run_with_GUI(initial_shift=initial_shift)
            else:
                # Run PMA algorithm
                shift = self.pma_object.run(initial_shift=initial_shift)
        except (Exception, KeyboardInterrupt) as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            shift = self.pma_object.total_shift * self.pma_object.scale
        finally:
            return shift


def load_task(
    file_path: str,
    exclude: Optional[str] = None,
    load_pma_sequence_volumes: bool = False,
) -> LaminographyAlignmentTask:
    print("Loading task from", file_path, "...")

    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    with h5py.File(file_path, "r") as h5_obj:
        # Load projections
        loaded_projections = load_ptycho_projections(h5_obj, exclude)

        # Insert projections into task along with saved task options
        task = LaminographyAlignmentTask(
            options=load_options_from_h5_group(h5_obj["options"], AlignmentTaskOptions),
            complex_projections=loaded_projections["complex_projections"],
            phase_projections=loaded_projections["phase_projections"],
        )

        # make sure all device options work on the current machine
        gpu_utils.auto_update_gpu_options(task.options)

        # Restore the PMA sequence if it was saved alongside the task.
        # Older task files won't have this group; falling through here
        # leaves task.pma_sequence as the empty sequence the constructor
        # already created.
        if "pma_sequence" in h5_obj:
            task.pma_sequence = PMASequence._load_from_group(
                h5_obj["pma_sequence"],
                include_volumes=load_pma_sequence_volumes,
            )

        print("Loading complete")

    return task
