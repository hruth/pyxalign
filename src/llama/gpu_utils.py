from array import ArrayType
from functools import wraps
from types import ModuleType
import cupy as cp
import scipy
import cupyx
import cupyx.scipy.fft as cufft
import numpy as np
from typing import Sequence, Union
import llama.api.enums as enums
from llama.api.options.device import DeviceOptions

from llama.api.types import ArrayType


def get_available_gpus():
    # Get the number of available GPU devices
    num_gpus = cp.cuda.runtime.getDeviceCount()

    # List all available GPU devices
    for i in range(num_gpus):
        device = cp.cuda.Device(i)
        print(f"GPU {i}: {device.name()}")


def pin_memory(array: np.ndarray):
    # Could use cupyx.empty_pinned instead to make it simpler..
    # Allocate pinned memory
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    # Create a new 1D array from an existing buffer
    # Just makes a array of zeros with the same data type and size as the buffer
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret


def is_pinned(array: ArrayType) -> bool:
    # Temporary -- this will only give the proper answer for large arrays
    min_array_size = 200
    if array.nbytes < min_array_size:
        raise NotImplementedError(
            f"This function does not work to check if arrays smaller than {min_array_size} bytes"
        )
    return array.nbytes > 2 * array.__sizeof__()


def move_to_device(
    array: Union[np.ndarray, cp.ndarray], device: enums.DeviceType, return_copy=False
) -> Union[np.ndarray, cp.ndarray]:
    if (device is enums.DeviceType.GPU and type(array) is cp.ndarray) or (
        device is enums.DeviceType.CPU and type(array) is np.ndarray
    ):
        if return_copy:
            return array * 1
        else:
            return array
    elif device is enums.DeviceType.GPU:
        return cp.array(array)
    elif device is enums.DeviceType.CPU:
        return array.get()


def get_fft_backend(array: ArrayType):
    module = cp.get_array_module(array)

    if module.__name__ == "numpy":
        fft_backend = "scipy"
    else:
        fft_backend = cufft

    return fft_backend


def get_scipy_module(array: ArrayType) -> ModuleType:
    module = cp.get_array_module(array)

    if module.__name__ == "numpy":
        scipy_module = scipy
    else:
        scipy_module = cupyx.scipy

    return scipy_module


def function_compute_device_manager(
    array_to_move_indices: Sequence[int] = [0],
    array_on_cpu_indices: Sequence[int] = [],
    single_array_input: Sequence[int] = [],
):
    """Wrapper for functions that have the option of being run on the CPU, the GPU, or multiple GPUs."""

    def inner_func(func):
        def wrapper(*args, **kwargs):
            # Device settings need to be passed in from the function kwargs
            # Would really like to find a better way of doing this
            if "device_options" not in kwargs.keys():
                # handle the default case here
                pass
            else:
                device_options = kwargs["device_options"]

            # Implementation to be added here

            func(*args, **kwargs)
            return wrapper

    return inner_func
