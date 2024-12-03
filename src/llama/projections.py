from typing import List, Optional
import numpy as np
import astra
import copy
from llama.api import enums, maps
from llama.api.options.alignment import AlignmentOptions
from llama.api.options.device import DeviceOptions

from llama.api.options.projections import ProjectionOptions
from llama.api.options.transform import ShiftOptions, UpsampleOptions
import llama.gpu_utils as gpu_utils
from llama.gpu_wrapper import device_handling_wrapper
from llama.mask import estimate_reliability_region_mask, blur_masks

import llama.plotting.plotters as plotters
from llama import reconstruct
from llama.transformations.classes import Downsampler, Shifter, Upsampler, Cropper
from llama.transformations.functions import image_shift_fft
from llama.unwrap import unwrap_phase
from llama.api.types import ArrayType


class Projections:
    def __init__(
        self,
        projections: np.ndarray,
        angles: np.ndarray,
        options: ProjectionOptions,
        masks: Optional[np.ndarray] = None,
        shift_manager: Optional["ShiftManager"] = None,
        center_of_rotation: Optional[np.ndarray] = None,
    ):
        self.options = options
        self.angles = angles

        # Crop and downsample input data if enabled
        self.data = Cropper(self.options.crop).run(projections)
        self.data = Downsampler(self.options.downsample).run(self.data)
        if masks is not None:
            self.masks = Cropper(self.options.crop).run(masks)
            mask_downsample_options = copy.deepcopy(self.options.downsample)
            self.options.downsample.type = self.options.mask_downsample_type
            self.masks = Downsampler(mask_downsample_options).run(self.masks)

        if shift_manager is not None:
            self.shift_manager = copy.deepcopy(shift_manager)
        else:
            self.shift_manager = ShiftManager(self.n_projections)

        if center_of_rotation is None:
            self.center_of_rotation = np.array(projections.shape[1:]) / 2

    @property
    def n_projections(self) -> int:
        return self.data.__len__()

    @property
    def reconstructed_object_dimensions(self) -> np.ndarray:
        laminography_angle = self.options.experiment.laminography_angle
        sample_thickness = self.options.experiment.sample_thickness
        pixel_size = self.options.experiment.pixel_size
        n_lateral_pixels = np.ceil(
            0.5 * self.data.shape[2] / np.cos(np.pi / 180 * (laminography_angle - 0.01))
        )
        n_pix = np.array([n_lateral_pixels, n_lateral_pixels, sample_thickness / pixel_size])
        n_pix = (np.ceil(n_pix / 32) * 32).astype(int)
        return n_pix

    def pin_projections(self):
        self.data = gpu_utils.pin_memory(self.data)

    def apply_staged_shift(self):
        self.shift_manager.apply_staged_shift(self)

    def plot_projections(self, process_function: callable = lambda x: x):
        plotters.make_image_slider_plot(process_function(self.data))

    def plot_shifted_projections(self, shift: np.ndarray, process_function: callable = lambda x: x):
        "Plot the shifted projections using the shift that was passed in."
        plotters.make_image_slider_plot(process_function(image_shift_fft(self.data, shift)))

    def plot_sum_of_projections(self, process_function: callable = lambda x: x):
        plotters.plot_sum_of_images(process_function(self.data))

    def get_masks(self, enable_plotting: bool = False):
        mask_options = self.options.mask
        downsample_options = self.options.mask.downsample
        if downsample_options.enabled:
            mask_options = copy.deepcopy(mask_options)
            scale = downsample_options.scale
            mask_options.binary_close_coefficient = mask_options.binary_close_coefficient / scale
            mask_options.binary_erode_coefficient = mask_options.binary_erode_coefficient / scale
            mask_options.fill = mask_options.fill / scale
        else:
            mask_options = mask_options
        # Calculate masks
        self.masks = estimate_reliability_region_mask(
            images=Downsampler(self.options.mask.downsample).run(self.data),
            options=mask_options,
            enable_plotting=enable_plotting,
        )
        # Upsample results
        upsample_options = UpsampleOptions(
            scale=downsample_options.scale, enabled=downsample_options.enabled
        )
        return Upsampler(upsample_options).run(self.masks)

    def blur_masks(self, kernel_sigma: int, use_gpu: bool = False):
        return blur_masks(self.masks, kernel_sigma, use_gpu)


class ComplexProjections(Projections):
    def unwrap_phase(self, pinned_results: Optional[np.ndarray] = None) -> ArrayType:
        unwrap_phase_wrapped = device_handling_wrapper(
            func=unwrap_phase,
            options=self.options.phase_unwrap.device,
            chunkable_inputs_for_gpu_idx=[0, 1],
            pinned_results=pinned_results,
        )
        return unwrap_phase_wrapped(self.data, self.masks, self.options.phase_unwrap)


class PhaseProjections(Projections):
    def get_3D_reconstruction(self, filter_inputs: bool = False):
        astra.set_gpu_index(self.options.astra.back_project_gpu_indices)
        scan_geometry_config, vectors = reconstruct.get_astra_reconstructor_geometry(
            sinogram=self.data,
            angles=self.angles,
            n_pix=self.reconstructed_object_dimensions,
            center_of_rotation=self.center_of_rotation,
            lamino_angle=self.options.experiment.laminography_angle,
            tilt_angle=self.options.experiment.tilt_angle,
            skew_angle=self.options.experiment.skew_angle,
        )
        astra_config = reconstruct.create_astra_reconstructor_config(
            self.data, scan_geometry_config, vectors
        )
        reconstruction = reconstruct.get_3D_reconstruction(astra_config)
        return reconstruction


class ShiftManager:
    # Might be better to attach this to the projections object
    # instead of the task object.
    # It might be useful to generalize this for tracking
    # any type of transformation.
    def __init__(self, n_projections: int):
        self.staged_shift = np.zeros((n_projections, 2))
        self.past_shifts: List[np.ndarray] = []
        self.past_shift_functions: List[enums.ShiftType] = []
        self.past_shift_options: List[AlignmentOptions] = []

    def stage_shift(
        self,
        shift: np.ndarray,
        function_type: enums.ShiftType,
    ):
        self.staged_shift = shift
        self.staged_function_type = function_type

    def unstage_shift(self):
        # Store staged values
        self.past_shifts += [self.staged_shift]
        self.past_shift_functions += [self.staged_function_type]
        # Clear the staged shift
        self.staged_shift = np.zeros_like(self.staged_shift)

    def apply_staged_shift(
        self,
        images: np.ndarray,
        device_options: DeviceOptions,
        pinned_results: Optional[np.ndarray],
    ):
        if self.is_shift_nonzero():
            shift_options = ShiftOptions(
                enabled=True, type=self.staged_function_type, device_options=device_options
            )
            images[:] = Shifter(shift_options).run(images, self.staged_shift, pinned_results)
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
