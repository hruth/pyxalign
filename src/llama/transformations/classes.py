from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

import llama.api.maps as maps
from llama.api.options.transform import (
    DownsampleOptions,
    PreProcessingOptions,
    ShiftOptions,
    TransformOptions,
)

from llama.api.types import ArrayType
from llama.gpu_wrapper import device_handling_wrapper


class Transformation(ABC):
    """`Transformation` objects run transformation functions using the passed in device options."""

    def __init__(self, options: TransformOptions):
        self.enabled = options.enabled
        self.options = options

    @abstractmethod
    def run(self, images: ArrayType, *args, **kwargs) -> ArrayType:
        pass


class Downsample(Transformation):
    def __init__(
        self,
        options: DownsampleOptions,
    ):
        super().__init__(options)
        self.options: DownsampleOptions = options

    def run(self, images: ArrayType, pinned_results: Optional[np.ndarray] = None) -> ArrayType:
        """Calls one of the image downsampling functions"""
        if self.enabled:
            self.function = device_handling_wrapper(
                func=maps.get_downsample_func_by_enum(self.options.type),
                options=self.options.device_options,
                chunkable_inputs_for_gpu_idx=...,
                pinned_results=pinned_results,
            )
            return self.function_type(images, self.options.scale)
        else:
            return images


class Shifter(Transformation):
    def __init__(
        self,
        options: ShiftOptions,
    ):
        super().__init__(options)
        self.options: ShiftOptions = options

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


class PreProcess(Transformation):
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
        images = Downsample(self.options.downsample_options).run(images)
        return images


