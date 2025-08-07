from typing import Optional
import numpy as np
import h5py
from PyQt5.QtWidgets import QApplication

from pyxalign import gpu_utils
from pyxalign.api.options.alignment import ProjectionMatchingOptions
from pyxalign.data_structures.projections import (
    ComplexProjections,
    PhaseProjections,
    Projections,
    get_kwargs_for_copying_to_new_projections_object,
)
from pyxalign.alignment.cross_correlation import CrossCorrelationAligner
from pyxalign.alignment.projection_matching import ProjectionMatchingAligner
from pyxalign.api.options.task import AlignmentTaskOptions
from pyxalign.api import enums
from pyxalign.api.types import r_type
from pyxalign.io.load import load_projections
from pyxalign.io.save import save_generic_data_structure_to_h5
from pyxalign.io.utils import load_options
from pyxalign.plotting.interactive.projection_matching import ProjectionMatchingViewer
from pyxalign.timing.timer_utils import clear_timer_globals
# from pyxalign.plotting.interactive.task import TaskViewer # causes circular imports


class LaminographyAlignmentTask:
    def __init__(
        self,
        options: AlignmentTaskOptions,
        complex_projections: Optional[ComplexProjections] = None,
        phase_projections: Optional[PhaseProjections] = None,
    ):
        self.options = options
        if phase_projections is None and complex_projections is None:
            raise Exception(
                "Projections must be included when creating an instance of LaminographyAlignmentTask"
            )

        self.complex_projections = complex_projections
        self.phase_projections = phase_projections
        self.pma_object: ProjectionMatchingAligner = None
        self.pma_gui_list: list[ProjectionMatchingViewer] = []

    def get_cross_correlation_shift(
        self,
        projection_type: enums.ProjectionType = enums.ProjectionType.COMPLEX,
        illum_sum: np.ndarray = None,
        plot_results: bool = True,
    ):
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
        )
        if plot_results:
            projections.plot_shift(
                shift_type=enums.ShiftManagerMemberType.STAGED_SHIFT,
                title="Cross-correlation Shift",
            )
        print("Cross-correlation shift stored in shift_manager")

    def get_projection_matching_shift(
        self, initial_shift: Optional[np.ndarray] = None
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

        # run the pma algorithm
        self.pma_object, shift = run_projection_matching(
            self.phase_projections, initial_shift, self.options.projection_matching
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

    def get_complex_projection_masks(self, enable_plotting: bool = False):
        clear_timer_globals()
        self.complex_projections.get_masks(enable_plotting)

    def get_unwrapped_phase(self, pinned_results: Optional[np.ndarray] = None):
        # if (
        #     self.phase_projections is not None
        #     and gpu_utils.is_pinned(self.phase_projections.data)
        #     and pinned_results is None
        # ):
        #     pinned_results = gpu_utils.pin_memory(self.phase_projections.data)
        # elif pinned_results is None:
        #     pinned_results = gpu_utils.create_empty_pinned_array(
        #         self.complex_projections.data.shape, dtype=r_type
        #     )
        if pinned_results is None:
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

    def save_task(self, file_path: str, exclude: list[str] = []):
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
                    projection.save_projections_object(h5_obj=h5_obj.create_group(attr))
            save_generic_data_structure_to_h5(self.options, h5_obj.create_group("options"))
            print(f"task saved to {h5_obj.file.filename}{h5_obj.name}")

    # def launch_viewer(self) -> QApplication:
    #     app = QApplication.instance() or QApplication([])
    #     self.gui = TaskViewer(self)
    #     self.gui.show()
    #     return app


def run_projection_matching(
    phase_projections: PhaseProjections,
    initial_shift: np.ndarray,
    projection_matching_options: ProjectionMatchingOptions,
) -> tuple[ProjectionMatchingAligner, np.ndarray]:
    # Initialize the projection-matching alignment object
    pma_object = ProjectionMatchingAligner(phase_projections, projection_matching_options)
    try:
        if pma_object.options.interactive_viewer.update.enabled:
            # Run PMA algorithm
            shift = pma_object.run_with_GUI(initial_shift=initial_shift)
        else:
            # Run PMA algorithm
            shift = pma_object.run(initial_shift=initial_shift)
    except (Exception, KeyboardInterrupt):
        shift = pma_object.total_shift * pma_object.scale
    finally:
        return pma_object, shift


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
