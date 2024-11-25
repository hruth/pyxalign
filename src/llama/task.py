from typing import List
import numpy as np
from llama.api.options.alignment import AlignmentOptions
from llama.projections import Projections
from llama.alignment.cross_correlation import CrossCorrelationAligner
from llama.api.options.task import AlignmentTaskOptions
from llama.api import enums
from llama.api import maps


class LaminographyAlignmentTask:
    def __init__(
        self, projections: Projections, options: AlignmentTaskOptions
    ):  # update later to allow complex or phase projections
        self.options = options
        self.projections = projections
        self.cross_correlation_aligner = CrossCorrelationAligner(
            self.projections, self.options.cross_correlation_options
        )
        self.shift_manager = ShiftManager(projections.n_projections)

    def get_cross_correlation_shift(self):
        self.illum_sum = np.ones_like(self.projections.data[0])  # Temporary placeholder
        shift = self.cross_correlation_aligner.run(self.illum_sum)
        self.shift_manager.stage_shift(
            shift, enums.ShiftType.CIRC, self.options.cross_correlation_options
        )
        print("Cross-correlation shift stored in shift_history")

    def apply_staged_shift(self):
        self.shift_manager.apply_staged_shift(self.projections)

    def get_complex_projection_masks(self, enable_plotting: bool = False):
        self.projections.get_masks(enable_plotting)


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

    def apply_staged_shift(self, projections: Projections):
        if self.is_shift_nonzero():
            image_shift_function = maps.get_shift_func_by_enum(self.staged_function_type)
            projections.data = image_shift_function(projections.data, self.staged_shift)
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
