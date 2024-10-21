from array import ArrayType
import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np
from typing import Union
import llama.api.enums as enums


def get_available_gpus():
    # Get the number of available GPU devices
    num_gpus = cp.cuda.runtime.getDeviceCount()

    # List all available GPU devices
    for i in range(num_gpus):
        device = cp.cuda.Device(i)
        print(f"GPU {i}: {device.name()}")


def pin_memory(array: np.ndarray):
    pass


def is_pinned(array: np.ndarray) -> bool:
    pass


def move_to_device(
    array: Union[np.ndarray, cp.ndarray], device: enums.DeviceType, return_copy=False) -> Union[np.ndarray, cp.ndarray]:
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
    

def get_array_module_and_fft_backend(array: ArrayType):
    module = cp.get_array_module(array)

    if module.__name__ == 'numpy':
        fft_backend = 'scipy'
    else:
        fft_backend = cufft

    return module, fft_backend