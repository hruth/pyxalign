from llama.api.enums import DownsampleType
import llama.transformations as transformations
from typing import Callable, Union
import numpy as np
import cupy as cp

ArrayType = Union[np.ndarray, cp.ndarray]


def get_downsample_func_by_enum(key: DownsampleType) -> Callable[[ArrayType, int], ArrayType]:
    return {
        DownsampleType.FFT: transformations.image_downsample_fft,
        DownsampleType.LINEAR: transformations.image_downsample_linear,
    }[key]
