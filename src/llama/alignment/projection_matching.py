import numpy as np
import cupy as cp
import copy
from typing import Union
from llama.alignment.base import Aligner
from llama.projections import PhaseProjections, Projections
from llama.api.options.projections import ProjectionDevice
from llama.transformations.classes import Cropper, Downsampler
from llama.transformations.functions import image_shift_fft
import llama.api.enums as enums
from llama.api.options.alignment import ProjectionMatchingOptions
import llama.gpu_utils as gutils
from llama.api.types import ArrayType, r_type, c_type

# For PMA, you will need to create a new projections object.
# That projections object should have methods for doing the FP and FBP, and maybe shifting too?

# 3 ways to run:
# 1) pin array and transfer to GPU many times
# 2) leave on GPU
# 3) leabe on CPU
# Option 1 requires pinning, option 2 does not require pinning, and neither does 3.
# Just implement option 2 for now. Preparation should be done before passing to
# the pma function. PMA function should be able to handle all by using the decorator
# within PMA functions and by settign astra functions.


class ProjectionMatchingAligner(Aligner):
    def __init__(self, projections: Projections, options: ProjectionMatchingOptions):
        super().__init__(projections, options)
        self.options: ProjectionMatchingOptions = self.options

    @gutils.memory_releasing_error_handler
    def run(self) -> np.ndarray:
        self.pma_projections = PhaseProjections(
            projections=self.projections.data * 1,
            angles=self.projections.angles * 1,
            options=copy.deepcopy(self.options.projections),
            masks=self.projections.masks * 1,
        )

        shift = self.calculate_alignment_shift()

        return shift

    # Notes about initial implementation:
    # - initial shift not supported
    # - only doing the pinned memory case first

    @gutils.memory_releasing_error_handler
    def calculate_alignment_shift(self):

        unshifted_masks, unshifted_projections = self.initialize()
        


    def initialize(self):
        if self.options.keep_on_gpu:
            unshifted_projections = cp.array(self.pma_projections.data)
            unshifted_masks = cp.array(self.pma_projections.masks)
        else:
            unshifted_projections = gutils.pin_memory(self.pma_projections.data)
            unshifted_masks = gutils.pin_memory(self.pma_projections.masks)

        self.total_shift = np.zeros((self.angles, 2), dtype=r_type)

        return unshifted_masks, unshifted_projections
