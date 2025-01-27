from typing import List, Optional
import numpy as np
import cupyx
import copy
from llama.api.options import transform
from llama.estimate_center import estimate_center_of_rotation, CenterOfRotationEstimateResults
from llama.api import enums
from llama.api.options.alignment import AlignmentOptions
from llama.api.options.device import DeviceOptions

from llama.api.options.projections import ProjectionOptions, ProjectionTransformOptions
from llama.api.options.transform import (
    CropOptions,
    RotationOptions,
    ShearOptions,
    ShiftOptions,
    UpsampleOptions,
    DownsampleOptions,
)
import llama.gpu_utils as gpu_utils
from llama.gpu_wrapper import device_handling_wrapper
from llama.laminogram import Laminogram
from llama.mask import estimate_reliability_region_mask, blur_masks

import llama.plotting.plotters as plotters
from llama.timing.timer_utils import timer
from llama.transformations.classes import Downsampler, Rotator, Shearer, Shifter, Upsampler, Cropper
from llama.transformations.functions import image_shift_fft, will_rotation_flip_aspect_ratio
from llama.unwrap import unwrap_phase
from llama.api.types import ArrayType, r_type


class TransformTracker:
    def __init__(
        self,
        rotation: r_type = 0,
        shear: r_type = 0,
        crop=None,
        downsample: int = 1,
    ):
        self.rotation = rotation
        self.shear = shear
        self.crop = crop
        self.downsample = downsample

    def update_rotation(self, angle: r_type):
        self.rotation += angle

    def update_shear(self, angle: r_type):
        self.shear += angle

    def update_crop(self, crop_options: CropOptions):
        pass

    def update_downsample(self, downsample: int):
        self.downsample *= downsample


class Projections:
    @timer()
    def __init__(
        self,
        projections: np.ndarray,
        angles: np.ndarray,
        options: ProjectionOptions,
        masks: Optional[np.ndarray] = None,
        shift_manager: Optional["ShiftManager"] = None,
        center_of_rotation: Optional[np.ndarray] = None,
        skip_pre_processing: bool = False,
        transform_tracker: Optional[TransformTracker] = None,
    ):
        self.options = options
        self.angles = angles
        self.data = projections
        self.masks = masks
        self.pixel_size = self.options.experiment.pixel_size * 1
        if transform_tracker is None:
            self.transform_tracker = TransformTracker()
        else:
            self.transform_tracker = transform_tracker
        if center_of_rotation is None:
            self.center_of_rotation = np.array(self.data.shape[1:], dtype=r_type) / 2
        else:
            self.center_of_rotation = np.array(center_of_rotation, dtype=r_type)

        if not skip_pre_processing:
            self.transform_projections(self.options.input_processing)

        if shift_manager is not None:
            self.shift_manager = copy.deepcopy(shift_manager)
        else:
            self.shift_manager = ShiftManager(self.n_projections)

        self._post_init()

    @timer()
    def transform_projections(
        self,
        options: ProjectionTransformOptions,
    ):
        # Transform the projections in the proper order
        if options.crop is not None:
            self.crop_projections(options.crop)
        if options.downsample is not None:
            self.downsample_projections(
                options.downsample,
                options.mask_downsample_type,
                options.mask_downsample_use_gaussian_filter,
            )
        if options.rotation is not None:
            self.rotate_projections(options.rotation)
        if options.shear is not None:
            self.shear_projections(options.shear)

    def crop_projections(self, options: CropOptions):
        if options.enabled:
            self.data = Cropper(options).run(self.data)
            if self.masks is not None:
                self.masks = Cropper(options).run(self.masks)
            self.transform_tracker.update_crop(options)
            # To do: insert code for updating center of rotation

    def downsample_projections(
        self,
        options: DownsampleOptions,
        mask_downsample_type: enums.DownsampleType,
        mask_downsample_use_gaussian_filter: bool,
    ):
        if options.enabled:
            self.data = Downsampler(options).run(self.data)
            if self.masks is not None:
                mask_options = copy.deepcopy(options)
                mask_options.type = mask_downsample_type
                mask_options.use_gaussian_filter = mask_downsample_use_gaussian_filter
                self.masks = Downsampler(mask_options).run(self.masks)
            self.transform_tracker.update_downsample(options.scale)
            # Update center of rotation
            self.center_of_rotation = self.center_of_rotation / options.scale
            # Update pixel size
            self.pixel_size = self.pixel_size * options.scale

    @timer()
    def rotate_projections(self, options: RotationOptions):
        # Note: the fourier rotation that is used for masks will likely
        # be an issue. A new method needs to be implemented.
        if options.enabled:
            data_aspect_ratio_changes = will_rotation_flip_aspect_ratio(options.angle)
            if not data_aspect_ratio_changes:
                Rotator(options).run(self.data, pinned_results=self.data)
                if self.masks is not None:
                    Rotator(options).run(self.masks, pinned_results=self.masks)
            else:
                # If projections are already pinned, create a new pinned array
                # at the new aspect ratio
                if gpu_utils.is_pinned(self.data):
                    # Note: for very large arrays where pinned memory
                    # needs to be carefully managed, this may cause
                    # problems since you are temporarily doubling the
                    # amount of pinned memory.
                    pinned_array = gpu_utils.create_empty_pinned_array(
                        shape=(self.n_projections, *self.size[::-1]),
                        dtype=self.data.dtype,
                    )
                    self.data = Rotator(options).run(self.data, pinned_results=pinned_array)
                else:
                    self.data = Rotator(options).run(self.data)
                # Do the same procedure for the masks
                if self.masks is not None:
                    if gpu_utils.is_pinned(self.masks):
                        pinned_array = gpu_utils.create_empty_pinned_array(
                            shape=(self.n_projections, *self.size[::-1]),
                            dtype=self.masks.dtype,
                        )
                        self.masks = Rotator(options).run(self.masks, pinned_results=pinned_array)
                    else:
                        self.masks = Rotator(options).run(self.masks)
            self.transform_tracker.update_rotation(options.angle)
            # To do: insert code for updating center of rotation

    @timer()
    def shear_projections(self, options: ShearOptions):
        if options.enabled:
            Shearer(options).run(self.data, pinned_results=self.data)
            if self.masks is not None:
                # Will probably be wrong without fixes
                self.masks = Shearer(options).run(self.masks)
            self.transform_tracker.update_shear(options.angle)
            # To do: insert code for updating center of rotation

    @property
    def n_projections(self) -> int:
        return self.data.__len__()

    @property
    def reconstructed_object_dimensions(self) -> np.ndarray:
        laminography_angle = self.options.experiment.laminography_angle
        sample_thickness = self.options.experiment.sample_thickness
        n_lateral_pixels = np.ceil(
            0.5 * self.data.shape[2] / np.cos(np.pi / 180 * (laminography_angle - 0.01))
        )
        n_pix = np.array([n_lateral_pixels, n_lateral_pixels, sample_thickness / self.pixel_size])
        n_pix = (np.ceil(n_pix / 32) * 32).astype(int)
        return n_pix

    @property
    def size(self):
        return self.data.shape[1:]

    def _post_init(self):
        "For running children specific code after instantiation"
        pass

    # def update_center_of_rotation(self):
    #     if self.options.downsample.enabled:
    #         self.center_of_rotation = self.center_of_rotation / self.options.downsample.scale
    #     if self.options.crop.enabled:
    #         # to do: add CoR update for when cropping is not symmetric. Will affect
    #         # how to handle downsampling CoR as well.
    #         pass

    def pin_projections(self):
        self.data = gpu_utils.pin_memory(self.data)

    def apply_staged_shift(self, device_options: Optional[DeviceOptions] = None):
        if device_options is None:
            device_options = DeviceOptions()
        self.shift_manager.apply_staged_shift(
            images=self.data,
            device_options=device_options,
            pinned_results=self.data,
        )

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
        # Keep on CPU for now -- can add upsampling device options later if needed
        upsample_options = UpsampleOptions(
            scale=downsample_options.scale, enabled=downsample_options.enabled
        )
        self.masks = Upsampler(upsample_options).run(self.masks)
        # return Upsampler(upsample_options).run(self.masks)

    def blur_masks(self, kernel_sigma: int, use_gpu: bool = False):
        return blur_masks(self.masks, kernel_sigma, use_gpu)


class ComplexProjections(Projections):
    def unwrap_phase(self, pinned_results: Optional[np.ndarray] = None) -> ArrayType:
        unwrap_phase_wrapped = device_handling_wrapper(
            func=unwrap_phase,
            options=self.options.phase_unwrap.device,
            chunkable_inputs_for_gpu_idx=[0, 1],
            pinned_results=pinned_results,
            display_progress_bar=True,
        )
        return unwrap_phase_wrapped(self.data, self.masks, self.options.phase_unwrap)


class PhaseProjections(Projections):
    def _post_init(self):
        self.laminogram = Laminogram(self)

    def get_3D_reconstruction(
        self, filter_inputs: bool = False, pinned_filtered_sinogram: Optional[np.ndarray] = None
    ):
        self.laminogram.generate_laminogram(
            filter_inputs=filter_inputs,
            pinned_filtered_sinogram=pinned_filtered_sinogram,
        )

    def estimate_center_of_rotation(self) -> CenterOfRotationEstimateResults:
        return estimate_center_of_rotation(
            self.data, self.angles, self.masks, self.options.estimate_center
        )


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
        # Clear the staged variables
        self.staged_shift = np.zeros_like(self.staged_shift)
        self.staged_function_type = None
        self.staged_alignment_options = None

    def apply_staged_shift(
        self,
        images: np.ndarray,
        device_options: Optional[DeviceOptions] = None,
        pinned_results: Optional[np.ndarray] = None,
    ):
        if self.is_shift_nonzero():
            shift_options = ShiftOptions(
                enabled=True,
                type=self.staged_function_type,
                device=device_options,
            )
            images[:] = Shifter(shift_options).run(
                images=images,
                shift=self.staged_shift,
                pinned_results=pinned_results,
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