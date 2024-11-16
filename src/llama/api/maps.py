from llama.api.enums import DownsampleType
import llama.transformations.functions
from typing import Callable, Union
import numpy as np
import cupy as cp

ArrayType = Union[np.ndarray, cp.ndarray]


def get_downsample_func_by_enum(key: DownsampleType) -> Callable[[ArrayType, int], ArrayType]:
    return {
        DownsampleType.FFT: llama.transformations.functions.image_downsample_fft,
        DownsampleType.LINEAR: llama.transformations.functions.image_downsample_linear,
    }[key]
