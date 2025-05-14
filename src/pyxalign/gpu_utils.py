from functools import wraps
import traceback
from types import ModuleType
from typing import Any, Callable, List, Optional, TypeVar
import cupy as cp
import scipy
import cupyx
import cupyx.scipy.signal
import cupyx.scipy.interpolate
import cupyx.scipy.fft as cufft
import numpy as np
import sys
from typing import Union
import pyxalign.api.enums as enums
# from pyxalign.timer import timer
import pyxalign.timing.timer_utils as timer_utils

from pyxalign.api.types import ArrayType

T = TypeVar("T", bound=Callable[..., Any])


def get_available_gpus() -> tuple[int]:
    return tuple(range(cp.cuda.runtime.getDeviceCount()))


def turn_off_fft_cache(gpu_indices: Optional[List[int]] = None):
    if gpu_indices is None:
        gpu_indices = get_available_gpus()
    for gpu in gpu_indices:
        with cp.cuda.Device(gpu):
            cp.fft.config.get_plan_cache().set_size(0)


def free_blocks_on_all_gpus(gpu_indices: Optional[List[int]] = None, show_info: bool = False):
    if gpu_indices is None:
        gpu_indices = get_available_gpus()
    for gpu in gpu_indices:
        with cp.cuda.Device(gpu):
            if show_info:
                print("Before freeing:")
                print_gpu_memory_use()
            cp.get_default_memory_pool().free_all_blocks()
            if show_info:
                print("After freeing:")
                print_gpu_memory_use()


def print_gpu_memory_use():
    bytes_to_MiB = 1 / 1048576
    mempool = cp.get_default_memory_pool()
    print(cp.cuda.Device())
    print("   Used:", round(mempool.used_bytes() * bytes_to_MiB), "MiB")
    print("  Total:", round(mempool.total_bytes() * bytes_to_MiB), "MiB")


def check_gpu_list(num_gpus: int, gpu_indices: List[int]):
    gpu_count = cp.cuda.runtime.getDeviceCount()
    if num_gpus > gpu_count:
        raise ValueError(
            f"The number of GPUs specified in options is {num_gpus}, but the actual device count is only {gpu_count}!"
        )
    if any([index > gpu_count - 1 for index in gpu_indices[:num_gpus]]):
        raise ValueError(
            f"The specified GPU indices {gpu_indices[:num_gpus]} have value(s) greater than the actual device count ({gpu_count})"
        )


@timer_utils.timer()
def pin_memory(array: np.ndarray, force_repin: bool = False) -> np.ndarray:
    # Could use cupyx.empty_pinned instead to make it simpler..
    if force_repin or not is_pinned(array):
        # Allocate pinned memory
        mem = cp.cuda.alloc_pinned_memory(array.nbytes)
        # Create a new 1D array from an existing buffer
        # Just makes a array of zeros with the same data type and size as the buffer
        ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
        ret[...] = array
        return ret
    else: 
        return array


@timer_utils.timer()
def create_empty_pinned_array(shape: tuple, dtype: type[float]):
    return cupyx.empty_pinned(shape=shape, dtype=dtype)


@timer_utils.timer()
def create_empty_pinned_array_like(array: ArrayType):
    return cupyx.empty_like_pinned(array)


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


def get_scipy_module(
    array: ArrayType,
) -> ModuleType:  #  Literal[scipy, cupyx.scipy]:#ModuleType:  # , submodule: enums.SciPySubmodules) -> ModuleType:
    module = cp.get_array_module(array)

    if module.__name__ == "numpy":
        scipy_module = scipy
    else:
        scipy_module = cupyx.scipy

    return scipy_module


def memory_releasing_error_handler(func, show_info: bool = False) -> T:
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt as ex:
            print(f"{type(ex).__name__}: Execution stopped by user")
            traceback.print_exc()
        except Exception as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
        finally:
            for gpu in get_available_gpus():
                with cp.cuda.Device(gpu):
                    if show_info:
                        print_gpu_memory_use()
                    cp.get_default_memory_pool().free_all_blocks()
                    if show_info:
                        print_gpu_memory_use()

    return wrapped
