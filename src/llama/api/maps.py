from typing import Callable, Union
import numpy as np
import cupy as cp

from llama.api.enums import DownsampleType, ShiftType
import llama.transformations.functions

from llama.api.types import ArrayType


class ShiftProtocol:
    def __call__(self, images: ArrayType, shift: int) -> ArrayType: ...


# To do: make a protocol to help with type hints
def get_downsample_func_by_enum(key: DownsampleType) -> Callable[[ArrayType, int], ArrayType]:
    return {
        DownsampleType.FFT: llama.transformations.functions.image_downsample_fft,
        DownsampleType.LINEAR: llama.transformations.functions.image_downsample_linear,
    }[key]


def get_shift_func_by_enum(key: DownsampleType) -> ShiftProtocol:
    return {
        ShiftType.CIRC: llama.transformations.functions.image_shift_circ,
        ShiftType.FFT: llama.transformations.functions.image_shift_fft,
    }[key]
