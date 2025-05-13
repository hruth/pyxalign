from typing import Optional
import numpy as np
import h5py
from PyQt5.QtWidgets import QApplication

from llama import gpu_utils
from llama.data_structures.projections import (
    ComplexProjections,
    PhaseProjections,
    Projections,
    get_kwargs_for_copying_to_new_projections_object,
)
from llama.alignment.cross_correlation import CrossCorrelationAligner
from llama.alignment.projection_matching import ProjectionMatchingAligner
from llama.api.options.task import AlignmentTaskOptions
from llama.api import enums
from llama.api.types import r_type
from llama.io.save import save_generic_data_structure_to_h5
from llama.plotting.interactive.projection_matching import ProjectionMatchingViewer
from llama.plotting.interactive.task import TaskViewer
from llama.timing.timer_utils import clear_timer_globals


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
        self.pma_GUI_list: list[ProjectionMatchingViewer] = []

    def get_cross_correlation_shift(
        self,
        projection_type: enums.ProjectionType = enums.ProjectionType.COMPLEX,
        illum_sum: np.ndarray = None,
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
        projections.plot_staged_shift("Cross-correlation Shift")
        print("Cross-correlation shift stored in shift_manager")

    def get_projection_matching_shift(self, initial_shift: Optional[np.ndarray] = None):
        if self.pma_object is not None:
            if hasattr(self.pma_object, "aligned_projections"):
                # Clear old astra objects
                self.pma_object.aligned_projections.laminogram.clear_astra_objects()

        clear_timer_globals()
        # Initialize the projection-matching alignment object
        self.pma_object = ProjectionMatchingAligner(
            self.phase_projections, self.options.projection_matching
        )
        if self.options.projection_matching.interactive_viewer.close_old_windows:
            self.clear_pma_GUI_list()
        try:
            if self.pma_object.options.interactive_viewer.update.enabled:
                # Run PMA algorithm
                shift = self.pma_object.run_with_GUI(initial_shift=initial_shift)
                # Store the QWidget in a list so the window remains open even if
                # another PMA loop is started
                self.pma_GUI_list += [self.pma_object.gui]  # uncomment later
                # Close window
                # self.pma_object.gui.close() # I think adding this helped, or removing the list helped.
            else:
                # Run PMA algorithm
                shift = self.pma_object.run(initial_shift=initial_shift)
        except (Exception, KeyboardInterrupt):
            shift = self.pma_object.total_shift * self.pma_object.scale
        finally:
            # Store the result in the ShiftManager object
            self.phase_projections.shift_manager.stage_shift(
                shift=shift,
                function_type=enums.ShiftType.FFT,
                alignment_options=self.options.projection_matching,
            )
            print("Projection-matching shift stored in shift_manager")

    def clear_pma_GUI_list(self):
        for gui in self.pma_GUI_list:
            gui.close()
        self.pma_GUI_list = []

    def get_complex_projection_masks(self, enable_plotting: bool = False):
        clear_timer_globals()
        self.complex_projections.get_masks(enable_plotting)

    def get_unwrapped_phase(self, pinned_results: Optional[np.ndarray] = None):
        if (
            self.phase_projections is not None
            and gpu_utils.is_pinned(self.phase_projections.data)
            and pinned_results is not None
        ):
            pinned_results = gpu_utils.pin_memory(self.phase_projections.data)
        elif pinned_results is None:
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

    def launch_viewer(self):
        app = QApplication.instance() or QApplication([])
        self.gui = TaskViewer(self)
        self.gui.show()
