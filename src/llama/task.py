from typing import Optional
import numpy as np
from llama.projections import ComplexProjections, PhaseProjections
from llama.alignment.cross_correlation import CrossCorrelationAligner
from llama.alignment.projection_matching import ProjectionMatchingAligner
from llama.api.options.task import AlignmentTaskOptions
from llama.api import enums
from llama.api.types import r_type
from llama.timing.timer_utils import clear_timer_globals


class LaminographyAlignmentTask:
    pma_object: ProjectionMatchingAligner = None

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

    def get_cross_correlation_shift(self):
        clear_timer_globals()
        # Only for complex projections for now
        # Does this really need to be saved as an attribute?
        self.cross_correlation_aligner = CrossCorrelationAligner(
            self.complex_projections, self.options.cross_correlation
        )
        # Placeholder for actual illum_sum
        self.illum_sum = np.ones_like(self.complex_projections.data[0], dtype=r_type)
        shift = self.cross_correlation_aligner.run(self.illum_sum)
        self.complex_projections.shift_manager.stage_shift(
            shift=shift,
            function_type=enums.ShiftType.CIRC,
            alignment_options=self.options.cross_correlation,
        )
        print("Cross-correlation shift stored in shift_manager")

    def get_projection_matching_shift(self, initial_shift: Optional[np.ndarray]=None):
        if self.pma_object is not None and hasattr(self.pma_object, "aligned_projections"):
            # Clear old astra objects
            self.pma_object.aligned_projections.laminogram.clear_astra_objects()

        clear_timer_globals()
        self.pma_object = ProjectionMatchingAligner(
            self.phase_projections, self.options.projection_matching
        )
        shift = self.pma_object.run(initial_shift=initial_shift)
        self.phase_projections.shift_manager.stage_shift(
            shift=shift,
            function_type=enums.ShiftType.FFT,
            alignment_options=self.options.projection_matching,
        )
        print("Projection-matching shift stored in shift_manager")

    def get_complex_projection_masks(self, enable_plotting: bool = False):
        clear_timer_globals()
        self.complex_projections.get_masks(enable_plotting)

    def get_unwrapped_phase(self, pinned_results: Optional[np.ndarray] = None):
        unwrapped_projections = self.complex_projections.unwrap_phase(pinned_results)
        self.phase_projections = PhaseProjections(
            unwrapped_projections,
            self.complex_projections.angles,
            self.complex_projections.options,
            self.complex_projections.masks,
            self.complex_projections.shift_manager,
            skip_pre_processing=True,
        )


