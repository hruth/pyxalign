import numpy as np
from typing import Union
from llama.projections import PhaseProjections
from llama.api.options.projections import ProjectionDeviceOptions
from llama.transformations import image_shift_fft
import llama.api.enums as enums
from src.llama.api.options.alignment import ProjectionMatchingOptions

# For PMA, you will need to create a new projections object.
# That projections object should have methods for doing the FP and FBP, and maybe shifting too?

class ProjectionMatchingAligner:
    def __init__(self):
        self.past_shifts = []

    def get_alignment_shift(self, projections: PhaseProjections, options: ProjectionMatchingOptions):
        self.options = options
        self.prepare_projections(projections.data)
        self.run_projection_matching_alignment()
        self.staged_shift = np.zeros((self.projections.n_projections, 2))

    def prepare_projections(self, projections: np.ndarray):
        # 3 ways to run:
        # 1) pin array and transfer to GPU many times
        # 2) leave on GPU
        # 3) leabe on CPU
        # Option 1 requires pinning, option 2 does not require pinning, and neither does 3.
        # Just implement option 2 for now. Preparation should be done before passing to
        # the pma function. PMA function should be able to handle all by using the decorator
        # within PMA functions and by settign astra functions.

        # Option 2)
        device_options = ProjectionDeviceOptions(pin_memory=False, device_type=enums.DeviceType.GPU)

        # Make the Projections object
        self.unshifted_projections = PhaseProjections(
            projections.data, self.options.downsampling, device_options
        )

    def run_projection_matching_alignment(self):
        pass

    # def apply_shift(self, projections, shift):
    #     return image_shift_fft(projections, shift)
