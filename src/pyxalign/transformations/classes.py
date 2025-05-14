from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from scipy import stats
from tqdm import tqdm
from pyxalign.api.enums import DownsampleType, DeviceType, ShiftType

import pyxalign.api.maps as maps
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.options.transform import (
    DownsampleOptions,
    PadOptions,
    ShiftOptions,
    TransformOptions,
    UpsampleOptions,
    CropOptions,
    ShearOptions,
    RotationOptions,
)

from pyxalign.api.types import ArrayType
from pyxalign.gpu_wrapper import device_handling_wrapper
from pyxalign.transformations.functions import image_crop, image_crop_pad
from pyxalign.timing.timer_utils import timer


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
        self,
        images: ArrayType,
        shift: np.ndarray,
        pinned_results: Optional[np.ndarray] = None,
        is_binary_mask: bool = False,
    ) -> ArrayType:
        """Calls one of the image shifting functions"""
        if self.enabled:
            self.function = device_handling_wrapper(
                func=maps.get_shift_func_by_enum(self.options.type),
                options=self.options.device,
                chunkable_inputs_for_gpu_idx=[0, 1],
                pinned_results=pinned_results,
            )

            if self.options.type == ShiftType.LINEAR:
                images = images * 1

            images = self.function(images, shift)

            if is_binary_mask and self.options.type == ShiftType.FFT:
                idx = images > 0.5
                images[:] = 0
                images[idx] = 1

            return images
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
    def run(self, images: ArrayType, pinned_results: Optional[np.ndarray] = None) -> ArrayType:
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
    def run(self, images: ArrayType, pinned_results: Optional[np.ndarray] = None) -> ArrayType:
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
            cropped_images = image_crop(
                images,
                horizontal_range,
                vertical_range,
                self.options.horizontal_offset,
                self.options.vertical_offset,
            )
            if self.options.return_view:
                return cropped_images
            else:
                return cropped_images * 1
        else:
            return images


class Padder(Transformation):
    def __init__(
        self,
        options: PadOptions,
    ):
        super().__init__(options)
        self.options: PadOptions = options

    @timer()
    def run(self, images: ArrayType) -> ArrayType:
        """Calls the image padding function"""
        if self.enabled:
            if self.options.new_extent_x is None:
                new_extent_x = images.shape[2]
            else:
                new_extent_x = self.options.new_extent_x
            if self.options.new_extent_y is None:
                new_extent_y = images.shape[1]
            else:
                new_extent_y = self.options.new_extent_y

            padded_images = np.zeros(
                (images.shape[0], new_extent_y, new_extent_x), dtype=images.dtype
            )
            for i in tqdm(range(len(images))):
                pad_value = self.options.pad_value
                padded_images[i] = image_crop_pad(
                    images[i], new_extent_y, new_extent_x, "constant", pad_value
                )

            return padded_images
        else:
            return images