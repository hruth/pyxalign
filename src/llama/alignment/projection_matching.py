import numpy as np
import cupy as cp
import scipy
import copy
from typing import Union
from llama.alignment.base import Aligner
from llama.projections import PhaseProjections, Projections
from llama.api.options.projections import ProjectionDevice
from llama.transformations.classes import Cropper, Downsampler
from llama.transformations.functions import image_shift_fft
import llama.image_processing as ip
import llama.api.enums as enums
import llama.api.maps as maps
from llama.api.enums import MemoryConfig
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
        self.aligned_projections = PhaseProjections(
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
        self.initialize_attributes()
        unshifted_masks, unshifted_projections = self.initialize_arrays()
        tukey_window, circulo = self.initialize_windows()

    def initialize_attributes(self):
        self.n_pix = self.aligned_projections.reconstructed_object_dimensions

    def initialize_arrays(self):
        self.memory_config = maps.get_memory_config_enum(
            self.options.keep_on_gpu, self.options.device_options.device_type
        )

        if self.memory_config is MemoryConfig.GPU_ONLY:
            initializer_function = cp.array
            self.xp = cp
            self.scipy_module: scipy = gutils.get_scipy_module(cp.array(1))
        elif self.memory_config is MemoryConfig.MIXED:
            initializer_function = gutils.pin_memory
            self.xp = cp
            self.scipy_module: scipy = gutils.get_scipy_module(cp.array(1))
        elif self.memory_config is MemoryConfig.CPU_ONLY:
            initializer_function = lambda x: (x * 1)  # noqa: E731
            self.xp = np
            self.scipy_module: scipy = gutils.get_scipy_module(np.array(1))

        unshifted_projections = initializer_function(self.aligned_projections.data)
        unshifted_masks = initializer_function(self.aligned_projections.masks)

        if self.memory_config is not MemoryConfig.CPU_ONLY:
            self.aligned_projections.data = initializer_function(self.aligned_projections.data)
            self.aligned_projections.masks = initializer_function(self.aligned_projections.masks)

        self.total_shift = self.xp.zeros((self.aligned_projections.angles, 2), dtype=r_type)

        return unshifted_masks, unshifted_projections

    def initialize_windows(self) -> tuple[ArrayType, ArrayType]:
        # Generate window for removing edge issues
        tukey_window = ip.get_tukey_window(A=0.2, scipy_module=self.scipy_module)
        # Generate circular mask for reconstruction
        circulo = ip.apply_3D_apodization(np.zeros(self.n_pix), 0, 5).astype(r_type)

        if self.memory_config is MemoryConfig.MIXED:
            tukey_window = gutils.pin_memory(tukey_window)

        return tukey_window, circulo
