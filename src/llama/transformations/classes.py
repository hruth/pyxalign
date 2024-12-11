from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from llama.api.enums import DownsampleType, UpsampleType

import llama.api.maps as maps
from llama.api.options.transform import (
    DownsampleOptions,
    PreProcessingOptions,
    ShiftOptions,
    TransformOptions,
    UpsampleOptions,
    CropOptions,
)

from llama.api.types import ArrayType
from llama.gpu_wrapper import device_handling_wrapper
from llama.transformations.functions import image_crop
from llama.timer import timer
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

    @timer("Downsampler")
    def run(
        self,
        images: ArrayType,
        shift: Optional[ArrayType] = None,
        pinned_results: Optional[np.ndarray] = None,
    ) -> ArrayType:
        """Calls one of the image downsampling functions"""
        # Note: currently the linear downsampling function also has the option to shift
        # the inputs.
        if self.enabled:
            if self.options.type is DownsampleType.LINEAR and shift is not None:
                self.function = device_handling_wrapper(
                    func=maps.get_downsample_func_by_enum(self.options.type),
                    options=self.options.device_options,
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
                    options=self.options.device_options,
                    chunkable_inputs_for_gpu_idx=[0],
                    pinned_results=pinned_results,
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

    @timer("Upsampler")
    def run(self, images: ArrayType, pinned_results: Optional[np.ndarray] = None) -> ArrayType:
        """Calls one of the image upsampling functions"""
        if self.enabled:
            self.function = device_handling_wrapper(
                func=maps.get_upsample_func_by_enum(self.options.type),
                options=self.options.device_options,
                chunkable_inputs_for_gpu_idx=[0],
                pinned_results=pinned_results,
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
    
    @timer("Shifter")
    def run(
        self, images: ArrayType, shift: np.ndarray, pinned_results: Optional[np.ndarray] = None
    ) -> ArrayType:
        """Calls one of the image shifting functions"""
        if self.enabled:
            self.function = device_handling_wrapper(
                func=maps.get_shift_func_by_enum(self.options.type),
                options=self.options.device_options,
                chunkable_inputs_for_gpu_idx=[0, 1],
                pinned_results=pinned_results,
            )
            return self.function(images, shift)
        else:
            return images


class Cropper(Transformation):
    def __init__(
        self,
        options: CropOptions,
    ):
        super().__init__(options)
        self.options: CropOptions = options

    def run(self, images: ArrayType) -> ArrayType:
        """Calls the image cropping function"""
        if self.enabled:
            return image_crop(
                images,
                self.options.horizontal_range,
                self.options.vertical_range,
                self.options.horizontal_offset,
                self.options.vertical_offset,
            )
        else:
            return images


class PreProcesser(Transformation):
    def __init__(
        self,
        options: PreProcessingOptions,
    ):
        super().__init__(options)
        self.options = options
        # self.enabled = options.enabled

    def run(self, images: ArrayType) -> ArrayType:
        # To add:
        # shift
        # crop
        images = Downsampler(self.options.downsample_options).run(images)
        return images
