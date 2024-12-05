import numpy as np
import cupy as cp
import scipy
import copy
from typing import Union
from llama.alignment.base import Aligner
from llama.api.options.projections import ProjectionOptions
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

# To do: 
# -add option for creating pinned arrays to speed up downsampling


class ProjectionMatchingAligner(Aligner):
    def __init__(self, projections: PhaseProjections, options: ProjectionMatchingOptions):
        super().__init__(projections, options)
        self.options: ProjectionMatchingOptions = self.options
        # self.options.reconstruct.filter.device = self.options.device

    @gutils.memory_releasing_error_handler
    def run(self) -> np.ndarray:
        # Create the projections object
        projection_options = copy.deepcopy(self.projections.options)
        projection_options.crop = self.options.crop
        projection_options.downsample = self.options.downsample
        projection_options.reconstruct = self.options.reconstruct
        self.aligned_projections = PhaseProjections(
            projections=self.projections.data,
            angles=self.projections.angles,
            options=projection_options,
            masks=self.projections.masks,
        )
        if self.options.downsample.enabled:
            self.scale = self.options.downsample.scale
        else:
            self.scale = 1
        # Make sure the reference doesn't go back to the original arrays
        if self.aligned_projections.data is self.projections.data:
            self.aligned_projections.data = self.aligned_projections.data * 1
        if self.aligned_projections.masks is self.projections.masks:
            self.aligned_projections.masks = self.aligned_projections.masks * 1

        # Run the PMA algorithm
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

        for self.iteration in range(self.options.iterations):
            print("Iteration: ", self.iteration)
            self.iterate(unshifted_projections, unshifted_masks, tukey_window, circulo)

    def iterate(self, unshifted_projections, unshifted_masks, tukey_window, circulo):
        "Execute an iteration of the projection-matching aligment loop"
        xp = self.xp

        self.apply_new_shift(unshifted_projections, unshifted_masks)
        # Apply tukey window filter - convert to gpu chunked later
        self.aligned_projections.masks[:] = tukey_window * self.aligned_projections.masks
        # Get back projection:
        # - use method from projections class and add filtering
        # - later, add ability to re-use the same astra_config
        self.reconstruction = self.aligned_projections.get_3D_reconstruction(
            filter_inputs=True, pinned_filtered_sinogram=self.pinned_filtered_sinogram
        )

    def apply_new_shift(self, unshifted_projections, unshifted_masks):
        if self.iteration != 0:
            shift_update = self.total_shift / self.scale
            self.projection_shifter.run(
                images=unshifted_projections,
                shift=shift_update,
                pinned_results=self.aligned_projections.data,
            )
            self.mask_shifter.run(
                images=unshifted_masks,
                shift=shift_update,
                pinned_results=self.aligned_projections.masks,
            )

    def initialize_attributes(self):
        xp = cp.get_array_module(self.aligned_projections.data)
        self.n_pix = self.aligned_projections.reconstructed_object_dimensions
        self.mass = xp.median(xp.abs(self.aligned_projections.data).mean(axis=(1, 2)))
        if self.memory_config is MemoryConfig.MIXED:
            self.pinned_filtered_sinogram = gutils.pin_memory(
                np.empty_like(self.aligned_projections.data)
            )
        else:
            self.pinned_filtered_sinogram = None

    def initialize_arrays(self):
        self.memory_config = maps.get_memory_config_enum(
            self.options.keep_on_gpu, self.options.device.device_type
        )

        if self.memory_config is MemoryConfig.GPU_ONLY:
            initializer_function = cp.array
            self.xp = cp
            self.scipy_module: scipy = gutils.get_scipy_module(cp.array(1))
        elif self.memory_config is MemoryConfig.MIXED:
            initializer_function = gutils.pin_memory
            self.xp = np
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

        self.total_shift = self.xp.zeros(
            (self.aligned_projections.angles.shape[0], 2), dtype=r_type
        )

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
        tukey_window = ip.get_tukey_window(self.aligned_projections.size, A=0.2)
        # Generate circular mask for reconstruction
        circulo = ip.apply_3D_apodization(np.zeros(self.n_pix), 0, 5).astype(r_type)

        if self.memory_config is MemoryConfig.MIXED:
            tukey_window = gutils.pin_memory(tukey_window)

        return tukey_window, circulo
