from typing import List, Optional
import numpy as np
import copy
from llama.api import enums, maps
from llama.api.options.alignment import AlignmentOptions

from llama.api.options.projections import ProjectionOptions
from llama.api.options.transform import UpsampleOptions
import llama.gpu_utils as gpu_utils
from llama.gpu_wrapper import device_handling_wrapper
from llama.mask import estimate_reliability_region_mask, blur_masks

import llama.plotting.plotters as plotters
from llama.transformations.classes import Downsampler, Upsampler
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
        shift_manager: Optional["ShiftManager"] = None
    ):
        self.data = projections
        self.options = options
        self.angles = angles
        if masks is not None:
            self.masks = masks
        if shift_manager is not None:
            self.shift_manager = copy.deepcopy(shift_manager)
        else:
            self.shift_manager = ShiftManager(self.n_projections)

        self.center_of_rotation = np.array(projections.shape[1:]) / 2

    @property
    def n_projections(self) -> int:
        return self.data.__len__()

    @property
    def reconstructed_object_dimensions(self) -> np.ndarray:
        # function for calculating n_pix_align
        pass

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def unwrap_phase(self, pinned_results: Optional[np.ndarray] = None) -> ArrayType:
        unwrap_phase_wrapped = device_handling_wrapper(
            func=unwrap_phase,
            options=self.options.phase_unwrap.device,
            chunkable_inputs_for_gpu_idx=[0, 1],
            pinned_results=pinned_results,
        )
        return unwrap_phase_wrapped(self.data, self.masks, self.options.phase_unwrap)


class PhaseProjections(Projections):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        # Clear the staged shift
        self.staged_shift = np.zeros_like(self.staged_shift)

    def apply_staged_shift(self, projections: Projections):
        if self.is_shift_nonzero():
            image_shift_function = maps.get_shift_func_by_enum(self.staged_function_type)
            projections.data = image_shift_function(
                projections.data, self.staged_shift
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
