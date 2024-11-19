from abc import ABC, abstractmethod
from functools import wraps
from typing import List
import numpy as np

import llama.api.maps as maps
from llama.api.options.device import DeviceOptions
from llama.api.options.transform import (
    DownsampleOptions,
    PreProcessingOptions,
    ShiftOptions,
    TransformOptions,
)

from llama.api.types import ArrayType


class Transformation(ABC):
    """`Transformation` objects run transformation functions using the passed in device options."""

    def __init__(self, options: TransformOptions):
        self.enabled = options.enabled
        self.options = options

    @abstractmethod
    def run(self, images: ArrayType, *args, **kwargs) -> ArrayType:
        pass

    def move_to_device(self, images: ArrayType):
        self.options.device_options


class Downsample(Transformation):
    def __init__(
        self,
        options: DownsampleOptions,
    ):
        super().__init__(options)
        self.scale = options.scale
        self.function_type = maps.get_downsample_func_by_enum(options.type)

    def run(self, images: ArrayType) -> ArrayType:
        """Calls one of the image downsampling functions"""
        if self.enabled:
            return self.function_type(images, self.scale)
        else:
            return images


class Shifter(Transformation):
    def __init__(
        self,
        options: ShiftOptions,
    ):
        super().__init__(options)
        self.function = device_handling_wrapper(
            func=maps.get_shift_func_by_enum(options.type),
            options=self.options.device_options,
            chunkable_inputs_gpu=[0],
            chunkable_inputs_cpu=[1],
        )

    def run(self, images: ArrayType, shift: np.ndarray) -> ArrayType:
        """Calls one of the image shifting functions"""
        if self.enabled:
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


def device_handling_wrapper(
    func: callable,
    options: DeviceOptions,
    chunkable_inputs_gpu: List[int] = List[0],
    chunkable_inputs_cpu: List[int] = List[1],
    common_inputs: List[int] = [],
):
    @wraps(func)
    def wrapped(*args, **kwargs):
        # To do:
        # - Test on GPU
        # - Make multi-GPU possible
        print("wrapped!")
        result = func(*args, **kwargs)

        return result

    return wrapped
