import numpy as np
import cupy as cp
import scipy
import copy
from typing import Union
from llama.alignment.base import Aligner
from llama.api.options.transform import ShiftOptions
from llama.projections import PhaseProjections, Projections
from llama.transformations.classes import Shifter
from llama.transformations.functions import image_shift_fft
import llama.image_processing as ip
import llama.api.enums as enums
import llama.api.maps as maps
from llama.api.enums import DeviceType, MemoryConfig
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
    def __init__(self, projections: PhaseProjections, options: ProjectionMatchingOptions):
        super().__init__(projections, options)
        self.options: ProjectionMatchingOptions = self.options

    @gutils.memory_releasing_error_handler
    def run(self) -> np.ndarray:
        self.aligned_projections = PhaseProjections(
            projections=self.projections.data,
            angles=self.projections.angles,
            options=copy.deepcopy(self.options.projections),
            masks=self.projections.masks,
        )
        # To do: add option for creating pinned arrays
        # to speed up downsampling
        shift = self.calculate_alignment_shift()

        return shift

    # Notes about initial implementation:
    # - initial shift not supported
    # - only doing the pinned memory case first

    @gutils.memory_releasing_error_handler
    def calculate_alignment_shift(self):
        unshifted_masks, unshifted_projections = self.initialize_arrays()
        self.initialize_attributes()
        self.initialize_shifters()
        tukey_window, circulo = self.initialize_windows()

        for self.iteration in range(self.options.iterations)[:3]:
            print("Iteration: ", self.iteration)
            self.iterate(unshifted_projections, unshifted_masks, tukey_window, circulo)

    def iterate(self, unshifted_projections, unshifted_masks, tukey_window, circulo):
        xp = self.xp

        self.apply_new_shift(unshifted_projections, unshifted_masks)
        # Apply tukey window filter - convert to gpu chunked later
        self.aligned_projections.masks[:] = tukey_window * self.aligned_projections.masks
        # calculate mass - look out for cp/np type issues
        mass = xp.median(xp.abs(self.aligned_projections.data).mean(axis=(1, 2)))
        # Get back projection:
        # - use method from projections class and add filtering
        # - later, add ability to re-use the same astra_config
        self.reconstruction = self.aligned_projections.get_3D_reconstruction(filter_inputs=True)

    def apply_new_shift(self, unshifted_projections, unshifted_masks):
        if self.iteration != 0:
            shift_update = self.total_shift / self.options.projections.downsample.scale
            self.projection_shifter.run(
                images=unshifted_projections,
                shift=shift_update,
                pinned_results=self.aligned_projections.data,
            )
            self.mask_shifter.run(
                iamges=unshifted_masks,
                shift=shift_update,
                pinned_results=self.aligned_projections.masks,
            )

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

        self.total_shift = self.xp.zeros((self.aligned_projections.angles.shape[0], 2), dtype=r_type)

        return unshifted_masks, unshifted_projections

    def initialize_shifters(self):
        device_options = copy.deepcopy(self.options.device)
        if self.memory_config is MemoryConfig.GPU_ONLY:
            device_options.device_type = DeviceType.GPU

        projections_shift_options = ShiftOptions(
            enabled=True,
            type=self.options.projection_shift_type,
            device_options=device_options,
        )
        mask_shift_options = ShiftOptions(
            enabled=True,
            type=self.options.mask_shift_type,
            device_options=device_options,
        )

        self.projection_shifter = Shifter(projections_shift_options)
        self.mask_shifter = Shifter(mask_shift_options)

    def initialize_windows(self) -> tuple[ArrayType, ArrayType]:
        # Generate window for removing edge issues
        tukey_window = ip.get_tukey_window(
            self.aligned_projections.size, A=0.2, scipy_module=self.scipy_module
        )
        # Generate circular mask for reconstruction
        circulo = ip.apply_3D_apodization(np.zeros(self.n_pix), 0, 5).astype(r_type)

        if self.memory_config is MemoryConfig.MIXED:
            tukey_window = gutils.pin_memory(tukey_window)

        return tukey_window, circulo
