from functools import wraps
from typing import List
import cupy as cp
import numpy as np

from llama.api.enums import DeviceType
from llama.api.options.device import DeviceOptions, GPUOptions

from llama.api.types import ArrayType


class InputArgumentsHandler:
    def __init__(
        self,
        args: tuple,
        options: GPUOptions,
        chunkable_inputs_gpu_idx: List[int],
        chunkable_inputs_cpu_idx: List[int],
        common_inputs_idx: List[int],
    ):
        self.list_ref = list(args)
        self.chunkable_inputs_gpu_idx = chunkable_inputs_gpu_idx
        self.chunkable_inputs_cpu_idx = chunkable_inputs_cpu_idx
        self.common_inputs_idx = common_inputs_idx

    @property
    def chunkable_inputs_gpu(self) -> List[ArrayType]:
        return [self.list_ref[i] for i in self.chunkable_inputs_gpu_idx]

    @property
    def common_inputs(self) -> List:
        return [self.list_ref[i] for i in self.common_inputs_idx]

    @property
    def chunkable_inputs_cpu(self) -> List[np.ndarray]:
        return [self.list_ref[i] for i in self.chunkable_inputs_cpu_idx]

    def set_chunkable_inputs_gpu(self, new_value: ArrayType, idx: int):
        self.list_ref[self.chunkable_inputs_gpu_idx[idx]] = new_value

    def set_chunkable_inputs_cpu(self, new_value: np.ndarray, idx: int):
        self.list_ref[self.chunkable_inputs_cpu_idx[idx]] = new_value

    def set_common_inputs(self, new_value, idx: int):
        self.list_ref[self.common_inputs_idx[idx]] = new_value

    def move_inputs_to_gpu(self):
        # single gpu case
        i = 0
        for input in self.chunkable_inputs_gpu:
            self.set_chunkable_inputs_gpu(cp.array(input), i)
            i += 1


def device_handling_wrapper(
    func: callable,
    options: DeviceOptions,
    chunkable_inputs_gpu_idx: List[int] = List[0],
    chunkable_inputs_cpu_idx: List[int] = [],
    common_inputs_idx: List[int] = [],
):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if options.device_type == DeviceType.CPU:
            result = func(*args, **kwargs)
            return result

        inputs = InputArgumentsHandler(
            args,
            options.gpu_options,
            chunkable_inputs_gpu_idx,
            chunkable_inputs_cpu_idx,
            common_inputs_idx,
        )
        keep_on_gpu = all(
            [cp.get_array_module(input) is cp for input in inputs.chunkable_inputs_gpu]
        )
        ### single gpu case ###
        inputs.move_inputs_to_gpu()
        result = func(*inputs.list_ref, **kwargs)

        return result

    return wrapped


def force_to_be_list(input_data) -> List:
    if isinstance(input_data, list):
        return input_data
    return [input_data]


if __name__ == "__main__":
    list_ref = [np.random.rand(10), np.random.rand(11), np.random.rand(12)]
    chunkable_inputs_gpu_idx = [0, 1]
    chunkable_inputs_cpu_idx = [2, 3, 4]
    common_inputs_idx = []
    inputs_list = InputArgumentsHandler(
        list_ref, chunkable_inputs_gpu_idx, chunkable_inputs_cpu_idx, common_inputs_idx
    )
    inputs_list.chunkable_inputs_gpu
