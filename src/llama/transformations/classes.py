from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from llama.api.enums import DownsampleType, UpsampleType, DeviceType, ShiftType

import llama.api.maps as maps
from llama.api.options.transform import (
    DownsampleOptions,
    ShiftOptions,
    TransformOptions,
    UpsampleOptions,
    CropOptions,
    ShearOptions,
    RotationOptions,
)

from llama.api.types import ArrayType
from llama.gpu_wrapper import device_handling_wrapper
from llama.transformations.functions import image_crop
from llama.timing.timer_utils import timer
# from llama.api import enums


class Transformation(ABC):
    """`Transformation` objects run transformation functions using the passed in device options."""

    def __init__(self, options: TransformOptions):
        self.enabled = options.enabled
        self.options = options

    @abstractmethod
    def run(self, images: ArrayType, *args, **kwargs) -> ArrayType:
        pass


class Downsampler(Transformation):
    def __init__(
        self,
        options: DownsampleOptions,
    ):
        super().__init__(options)
        self.options: DownsampleOptions = options

    @timer()
    def run(
        self,
        images: ArrayType,
        shift: Optional[ArrayType] = None,
        pinned_results: Optional[np.ndarray] = None,
    ) -> ArrayType:
        """Calls one of the image downsampling functions"""
        # Note: currently the linear downsampling function also has the option to shift
        # the inputs.
        if self.enabled and self.options.scale != 1:
            if self.options.type is DownsampleType.LINEAR and shift is not None:
                self.function = device_handling_wrapper(
                    func=maps.get_downsample_func_by_enum(self.options.type),
                    options=self.options.device,
                    chunkable_inputs_for_gpu_idx=[0, 2],
                    pinned_results=pinned_results,
                )
                return self.function(
                    images,
                    self.options.scale,
                    shift,
                    use_gaussian_filter=self.options.use_gaussian_filter,
                )
            else:
                self.function = device_handling_wrapper(
                    func=maps.get_downsample_func_by_enum(self.options.type),
                    options=self.options.device,
                    chunkable_inputs_for_gpu_idx=[0],
                    pinned_results=pinned_results,
                    display_progress_bar=True,
                )
                return self.function(
                    images,
                    self.options.scale,
                    use_gaussian_filter=self.options.use_gaussian_filter,
                )
        else:
            return images


class Upsampler(Transformation):
    def __init__(
        self,
        options: UpsampleOptions,
    ):
        super().__init__(options)
        self.options: UpsampleOptions = options

    @timer()
    def run(self, images: ArrayType, pinned_results: Optional[np.ndarray] = None) -> ArrayType:
        """Calls one of the image upsampling functions"""
        if self.enabled:
            self.function = device_handling_wrapper(
                func=maps.get_upsample_func_by_enum(self.options.type),
                options=self.options.device,
                chunkable_inputs_for_gpu_idx=[0],
                pinned_results=pinned_results,
                display_progress_bar=True,
            )
            return self.function(images, self.options.scale)
        else:
            return images


class Shifter(Transformation):
    def __init__(
        self,
        options: ShiftOptions,
    ):
        super().__init__(options)
        self.options: ShiftOptions = options
    
    @timer()
    def run(
        self, images: ArrayType, shift: np.ndarray, pinned_results: Optional[np.ndarray] = None, is_binary_mask: bool = False,
    ) -> ArrayType:
        """Calls one of the image shifting functions"""
        if self.enabled:
            self.function = device_handling_wrapper(
                func=maps.get_shift_func_by_enum(self.options.type),
                options=self.options.device,
                chunkable_inputs_for_gpu_idx=[0, 1],
                pinned_results=pinned_results,
            )

            images = self.function(images, shift)

            if is_binary_mask and self.options.type == ShiftType.FFT:
                idx = images > 0.5
                images[:] = 0
                images[idx] = 1

            return images
            # return self.function(images, shift)
        else:
            return images


class Rotator(Transformation):
    def __init__(
        self,
        options: RotationOptions,
    ):
        super().__init__(options)
        self.options: RotationOptions = options

    @timer()
    def run(
        self, images: ArrayType, pinned_results: Optional[np.ndarray] = None
    ) -> ArrayType:
        """Calls one of the image rotation functions"""
        if self.enabled:
            if self.options.device.device_type is DeviceType.CPU:
                raise NotImplementedError("This function is not supported on CPU.")
            self.function = device_handling_wrapper(
                func=maps.get_rotation_func_by_enum(self.options.type),
                options=self.options.device,
                chunkable_inputs_for_gpu_idx=[0],
                pinned_results=pinned_results,
                display_progress_bar=True,
            )
            return self.function(images, self.options.angle)
        else:
            return images


class Shearer(Transformation):
    def __init__(
        self,
        options: ShearOptions,
    ):
        super().__init__(options)
        self.options: ShearOptions = options

    @timer()
    def run(
        self, images: ArrayType, pinned_results: Optional[np.ndarray] = None
    ) -> ArrayType:
        """Calls one of the image shearing functions"""
        if self.enabled:
            self.function = device_handling_wrapper(
                func=maps.get_shear_func_by_enum(self.options.type),
                options=self.options.device,
                chunkable_inputs_for_gpu_idx=[0],
                pinned_results=pinned_results,
                display_progress_bar=True,
            )
            return self.function(images, self.options.angle)
        else:
            return images


class Cropper(Transformation):
    def __init__(
        self,
        options: CropOptions,
    ):
        super().__init__(options)
        self.options: CropOptions = options

    @timer()
    def run(self, images: ArrayType) -> ArrayType:
        """Calls the image cropping function"""
        if self.enabled:
            if self.options.horizontal_range == 0:
                horizontal_range = images.shape[2]
            else:
                horizontal_range = self.options.horizontal_range
            if self.options.vertical_range == 0:
                vertical_range = images.shape[1]
            else:
                vertical_range = self.options.vertical_range
            return 1 * image_crop(
                images,
                horizontal_range,
                vertical_range,
                self.options.horizontal_offset,
                self.options.vertical_offset,
            )
        else:
            return images