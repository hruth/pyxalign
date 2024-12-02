from typing import List, Optional
import numpy as np
from llama.api.options.alignment import AlignmentOptions
from llama.projections import Projections, ComplexProjections, PhaseProjections
from llama.alignment.cross_correlation import CrossCorrelationAligner
from llama.api.options.task import AlignmentTaskOptions
from llama.api import enums, maps
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
        self.cross_correlation_aligner = CrossCorrelationAligner(
            self.complex_projections, self.options.cross_correlation
        )
        self.shift_manager = ShiftManager(complex_projections.n_projections)

    def get_cross_correlation_shift(self):
        # Placeholder for actual illum_sum
        self.illum_sum = np.ones_like(self.complex_projections.data[0], dtype=r_type)
        shift = self.cross_correlation_aligner.run(self.illum_sum)
        self.shift_manager.stage_shift(shift, enums.ShiftType.CIRC, self.options.cross_correlation)
        print("Cross-correlation shift stored in shift_history")

    def apply_staged_shift(self):
        self.shift_manager.apply_staged_shift(self.complex_projections)

    def get_complex_projection_masks(self, enable_plotting: bool = False):
        self.complex_projections.get_masks(enable_plotting)

    def get_unwrapped_phase(self, pinned_results: Optional[np.ndarray] = None):
        unwrapped_projections = self.complex_projections.unwrap_phase(pinned_results)
        self.phase_projections = PhaseProjections(
            unwrapped_projections,
            self.complex_projections.angles,
            self.complex_projections.options,
            self.complex_projections.masks,
        )


class ShiftManager:
    def __init__(self, n_projections: int):
        self.staged_shift = np.zeros((n_projections, 2))
        self.past_shifts: List[np.ndarray] = []
        self.past_shift_functions: List[enums.ShiftType] = []
        self.past_shift_options: List[AlignmentOptions] = []

    def stage_shift(
        self,
        shift: np.ndarray,
        function_type: enums.ShiftType,
        alignment_options: AlignmentOptions,
    ):
        self.staged_shift = shift
        self.staged_function_type = function_type
        self.staged_alignment_options = alignment_options

    def unstage_shift(self):
        # Store staged values
        self.past_shifts += [self.staged_shift]
        self.past_shift_functions += [self.staged_function_type]
        self.past_shift_options += [self.staged_alignment_options]
        # Clear the staged shift
        self.staged_shift = np.zeros_like(self.staged_shift)

    def apply_staged_shift(self, complex_projections: Projections):
        if self.is_shift_nonzero():
            image_shift_function = maps.get_shift_func_by_enum(self.staged_function_type)
            complex_projections.data = image_shift_function(
                complex_projections.data, self.staged_shift
            )
            self.unstage_shift()
        else:
            print("There is no shift to apply!")

    def is_shift_nonzero(self):
        if self.staged_function_type is enums.ShiftType.CIRC:
            shift = np.round(self.staged_shift)
        else:
            shift = self.staged_shift
        if np.any(shift != 0):
            return True
        else:
            return False
