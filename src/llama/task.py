from typing import Optional
import numpy as np
from llama.projections import ComplexProjections, PhaseProjections
from llama.alignment.cross_correlation import CrossCorrelationAligner
from llama.api.options.task import AlignmentTaskOptions
from llama.api import enums
from llama.api.types import r_type


class LaminographyAlignmentTask:
    def __init__(
        self,
        options: AlignmentTaskOptions,
        complex_projections: Optional[ComplexProjections] = None,
        phase_projections: Optional[PhaseProjections] = None,
    ):
        self.options = options
        if complex_projections is not None:
            self.complex_projections = complex_projections
        if phase_projections is not None:
            self.phase_projections = phase_projections
        if phase_projections is None and complex_projections is None:
            raise Exception(
                "Projections must be included when creating an instance of LaminographyAlignmentTask"
            )

    def get_cross_correlation_shift(self):
        # Only for complex projections for now
        # Does this really need to be saved as an attribute?
        self.cross_correlation_aligner = CrossCorrelationAligner(
            self.complex_projections, self.options.cross_correlation
        )
        # Placeholder for actual illum_sum
        self.illum_sum = np.ones_like(self.complex_projections.data[0], dtype=r_type)
        shift = self.cross_correlation_aligner.run(self.illum_sum)
        self.complex_projections.shift_manager.stage_shift(
            shift, enums.ShiftType.CIRC, self.options.cross_correlation
        )
        print("Cross-correlation shift stored in shift_history")

    def get_complex_projection_masks(self, enable_plotting: bool = False):
        self.complex_projections.get_masks(enable_plotting)

    def get_unwrapped_phase(self, pinned_results: Optional[np.ndarray] = None):
        unwrapped_projections = self.complex_projections.unwrap_phase(pinned_results)
        self.phase_projections = PhaseProjections(
            unwrapped_projections,
            self.complex_projections.angles,
            self.complex_projections.options,
            self.complex_projections.masks,
            self.complex_projections.shift_manager,
        )


