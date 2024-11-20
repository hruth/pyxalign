from functools import wraps
from typing import List, Union
import cupy as cp
import numpy as np

from llama.api.enums import DeviceType
from llama.api.options.device import DeviceOptions, GPUOptions

from llama.api.types import ArrayType

real_dtype = np.float32
complex_dtype = np.complex64


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
        self.options = options
        self.n_chunks = int(
            np.ceil(len(self.chunkable_inputs_gpu[0]) / options.chunk_size)
        )
        self.initialize_chunked_list_ref()

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

    def initialize_chunked_list_ref(self):
        self.chunked_list_ref = [None for _ in range(len(self.list_ref))]
        for i in self.common_inputs_idx:
            self.chunked_list_ref[i] = self.list_ref[i]

    def move_inputs_to_gpu(self):
        # single gpu, single chunk
        i = 0
        for input in self.chunkable_inputs_gpu:
            self.set_chunkable_inputs_gpu(cp.array(input), i)
            i += 1

    def update_chunked_list(self, iter: int):
        # Return the list containing chunked versions of what is in list_ref
        idx_start, idx_stop = get_chunk_indices(iter, self.options.chunk_size)
        # Insert chunkable arguments
        for i in self.chunkable_inputs_gpu_idx:
            self.chunked_list_ref[i] = cp.array(self.list_ref[i][idx_start:idx_stop])
        for i in self.chunkable_inputs_cpu_idx:
            self.chunked_list_ref[i] = self.list_ref[i][idx_start:idx_stop]


class OutputResultsHandler:
    def __init__(
        self, options: GPUOptions, output_array_length: int, keep_on_gpu: bool
    ):
        self.options = options
        self.output_array_length = output_array_length
        self.keep_on_gpu = keep_on_gpu

    def update_results(self, chunked_results: Union[tuple, cp.ndarray], iter: int):
        # need to update to work with tuples later
        # if not isinstance(chunked_results, tuple):
        # chunked_results = (chunked_results,)
        if iter == 0:
            self.initialize_full_results(chunked_results)
        self.insert_into_full_results(chunked_results, iter)

    def insert_into_full_results(
        self, chunked_results: Union[tuple, cp.ndarray], iter: int
    ):
        idx_start, idx_stop = get_chunk_indices(iter, self.options.chunk_size)
        if self.keep_on_gpu:
            self.full_results[idx_start:idx_stop] = chunked_results
        else:
            chunked_results.get(out=self.full_results[idx_start:idx_stop])

    def initialize_full_results(self, chunked_results: Union[tuple, cp.ndarray]):
        output_array_size = (self.output_array_length, *chunked_results.shape[1:])
        # Need to modify this for dealing with pinned arrays being passed in!
        if self.keep_on_gpu:
            self.full_results = cp.empty(output_array_size, dtype=chunked_results.dtype)
        else:
            self.full_results = np.empty(output_array_size, dtype=chunked_results.dtype)

    # # Insert partial results into final results
    # for i in range(len(partialResults)):
    #     if keepOnGPU:
    #         # Insert the chunk of output arguments from this iteration
    #         # into the results array
    #         fullResults[i][inputStart:inputEnd] = partialResults[i]
    #     else:
    #         partialResults[i].get(out=fullResults[i][inputStart:inputEnd])


def get_chunk_indices(iter, chunk_size) -> tuple[int, int]:
    idx_start, idx_stop = iter * chunk_size, (iter + 1) * chunk_size
    return idx_start, idx_stop


def device_handling_wrapper(
    func: callable,
    options: DeviceOptions,
    chunkable_inputs_gpu_idx: List[int] = List[0],
    chunkable_inputs_cpu_idx: List[int] = [],
    common_inputs_idx: List[int] = [],
    pinned_results: List = [],
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

        if not options.gpu_options.chunking_enabled:
            inputs.move_inputs_to_gpu()
            result = func(*inputs.list_ref, **kwargs)
            return result

        outputs = OutputResultsHandler(
            options.gpu_options, inputs.chunkable_inputs_gpu[0].shape[0], keep_on_gpu
        )

        ### single gpu case ###
        for iter in range(inputs.n_chunks):
            # inputs.move_inputs_to_gpu()
            inputs.update_chunked_list(iter)
            chunked_results = func(*inputs.chunked_list_ref, **kwargs)
            outputs.update_results(chunked_results, iter)

        # # final processing
        # if len(results) == 1:
        #     results = results[0]

        return outputs.full_results
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
