from typing import Callable, Union
import numpy as np
import cupy as cp

from llama.api.enums import (
    DownsampleType,
    RotationType,
    ShiftType,
    UpsampleType,
    DeviceType,
    MemoryConfig,
)
import llama.transformations.functions
import llama.gpu_utils as gutils

from llama.api.types import ArrayType


class ShiftProtocol:
    def __call__(self, images: ArrayType, shift: int) -> ArrayType: ...


class RotationProtocol:
    def __call__(self, images: ArrayType, angle: int) -> ArrayType: ...


# Functions with enum inputs
# To do: make a protocol to help with type hints
def get_downsample_func_by_enum(key: DownsampleType) -> Callable:
    return {
        DownsampleType.FFT: llama.transformations.functions.image_downsample_fft,
        DownsampleType.LINEAR: llama.transformations.functions.image_downsample_linear,
        DownsampleType.NEAREST: llama.transformations.functions.image_downsample_nearest,
    }[key]


def get_upsample_func_by_enum(key: UpsampleType) -> Callable:
    return {
        UpsampleType.NEAREST: llama.transformations.functions.image_upsample_nearest,
    }[key]


def get_shift_func_by_enum(key: DownsampleType) -> ShiftProtocol:
    return {
        ShiftType.CIRC: llama.transformations.functions.image_shift_circ,
        ShiftType.FFT: llama.transformations.functions.image_shift_fft,
        ShiftType.LINEAR: llama.transformations.functions.image_shift_linear,
    }[key]


def get_rotation_func_by_enum(key: RotationType) -> RotationProtocol:
    return {
        RotationType.FFT: llama.transformations.functions.image_rotate_fft,
    }[key]


def get_shear_func_by_enum(key: RotationType) -> RotationProtocol:
    return {
        RotationType.FFT: llama.transformations.functions.image_shear_fft,
    }[key]


# Functions with enum outputs
def get_memory_config_enum(keep_on_gpu: bool, device_type: DeviceType):
    if keep_on_gpu:
        return MemoryConfig.GPU_ONLY
    elif device_type == DeviceType.GPU:
        return MemoryConfig.MIXED
    else:
        return MemoryConfig.CPU_ONLY
