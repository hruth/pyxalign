from functools import wraps
from typing import List, Optional, Union
import cupy as cp
import numpy as np
from llama import gpu_utils

from llama.api.enums import DeviceType
from llama.api.options.device import DeviceOptions, GPUOptions

from llama.api.types import ArrayType, r_type, c_type


class InputArgumentsHandler:
    def __init__(
        self,
        args: tuple,
        options: GPUOptions,
        chunkable_inputs_gpu_idx: List[int],
        chunkable_inputs_cpu_idx: List[int],
        common_inputs_gpu_idx: List[int],
    ):
        self.input_args = list(args)
        # self.input_args = args
        self.chunkable_inputs_gpu_idx = chunkable_inputs_gpu_idx
        self.chunkable_inputs_cpu_idx = chunkable_inputs_cpu_idx
        self.common_inputs_gpu_idx = common_inputs_gpu_idx
        self.options = options
        self.n_chunks = int(np.ceil(len(self.chunkable_inputs_gpu[0]) / options.chunk_size))
        self.initialize_devices()
        self.initialize_chunked_input_args()

    @property
    def chunkable_inputs_gpu(self) -> List[ArrayType]:
        return [self.input_args[i] for i in self.chunkable_inputs_gpu_idx]

    def set_chunkable_inputs_gpu(self, new_value: ArrayType, idx: int):
        self.input_args[self.chunkable_inputs_gpu_idx[idx]] = new_value

    def initialize_devices(self):
        gpu_utils.check_gpu_list(self.options.n_gpus, self.options.gpu_indices)
        self.gpu_list = self.options.gpu_indices[: self.options.n_gpus]

    def move_inputs_to_gpu_in_one_chunk(self):
        # single gpu, single chunk
        i = 0
        for input in self.chunkable_inputs_gpu:
            self.set_chunkable_inputs_gpu(cp.array(input), i)
            i += 1

    # def initialize_chunked_input_args(self):
    #     # This needs to be updated for multi-gpu!
    #     # self.chunked_input_args = [None for _ in range(len(self.input_args))]
    #     self.chunked_input_args = list(self.input_args)
    #     # Dump entire common inputs in the list of input args
    #     for i in self.common_inputs_gpu_idx:
    #         self.chunked_input_args[i] = cp.array(self.input_args[i])

    # def update_chunked_list(self, iter: int):
    #     # update chunked_list with chunked versions of what is in input_args
    #     idx_start, idx_stop = get_chunk_indices(iter, self.options.chunk_size)
    #     # Insert chunkable arguments
    #     for i in self.chunkable_inputs_gpu_idx:
    #         self.chunked_input_args[i] = cp.array(
    #             self.input_args[i][idx_start:idx_stop]
    #         )
    #     for i in self.chunkable_inputs_cpu_idx:
    #         self.chunked_input_args[i] = self.input_args[i][idx_start:idx_stop]

    def initialize_chunked_input_args(self):
        # self.chunked_input_args = [None for _ in range(len(self.input_args))]
        self.chunked_input_args = list(self.input_args)
        self.gpu_stager = GPUStager(
            self.gpu_list, self.options.chunk_size, self.chunkable_inputs_gpu
        )

    def update_chunked_list(self, iter: int, gpu_idx: int):
        # update chunked_list with chunked versions of what is in input_args
        idx_start, idx_stop = get_chunk_indices(iter, self.options.chunk_size)
        self.gpu_stager.update_chunked_inputs(gpu_idx, iter) # stage the next chunk on the gpu
        # Insert chunkable arguments
        for i in range(len(self.chunkable_inputs_gpu)):
            self.chunked_input_args[i] = self.gpu_stager.get_next_chunked_inputs(gpu_idx, i)
        for i in self.chunkable_inputs_cpu_idx:
            self.chunked_input_args[i] = self.input_args[i][idx_start:idx_stop]


class GPUStager:
    def __init__(self, gpu_list: List[int], chunk_size: int, chunkable_inputs_gpu: List[ArrayType]):#, chunkable_inputs_gpu_idx: List[int]):
        self.gpu_list = gpu_list
        self.chunkable_inputs_gpu = chunkable_inputs_gpu
        self.chunk_size = chunk_size
        self.initialize_list_of_arrays()

    @property
    def n_gpus(self):
        return len(self.gpu_list)

    def initialize_list_of_arrays(self):
        self.list_of_arrays = []
        for gpu in self.gpu_list:
            with cp.cuda.Device(gpu).use():
                self.list_of_arrays += [self.create_gpu_containers()]

    def create_gpu_containers(self):
        return [self.create_container_element(array) for array in self.chunkable_inputs_gpu]

    def create_container_element(self, array: ArrayType):
        array_size = (self.chunk_size, *array.shape[1:])
        return cp.empty(array_size, dtype=array.dtype)

    def update_chunked_inputs(self, gpu_idx: int, iter: int):
        "Insert data from full array into GPU chunk array"
        idx_start, idx_stop = get_chunk_indices(iter, self.chunk_size)
        # self.list_of_arrays[gpu_idx] = [
        #     cp.array(input) for input in self.chunkable_inputs_gpu[idx_start:idx_stop]
        # ]
        # self.list_of_arrays[gpu_idx].set(self.chunkable_inputs_gpu[idx_start:idx_stop])
        for i in range(len(self.chunkable_inputs_gpu)):
            self.list_of_arrays[gpu_idx][i].set(self.chunkable_inputs_gpu[i][idx_start:idx_stop])

    def get_next_chunked_inputs(self, gpu_idx: int, list_idx: int) -> List[cp.ndarray]:
        return self.list_of_arrays[gpu_idx][list_idx]


class OutputResultsHandler:
    def __init__(
        self,
        options: GPUOptions,
        output_array_length: int,
        keep_on_gpu: bool,
        pinned_results: Optional[np.ndarray] = None,
    ):
        self.options = options
        self.output_array_length = output_array_length
        self.keep_on_gpu = keep_on_gpu
        self.pinned_results = pinned_results

    def update_results(self, chunked_results: Union[tuple, cp.ndarray], iter: int):
        # need to update to work with tuples later
        # if not isinstance(chunked_results, tuple):
        # chunked_results = (chunked_results,)
        if iter == 0:
            self.initialize_full_results(chunked_results)
        self.insert_into_full_results(chunked_results, iter)

    def insert_into_full_results(self, chunked_results: Union[tuple, cp.ndarray], iter: int):
        idx_start, idx_stop = get_chunk_indices(iter, self.options.chunk_size)
        if self.keep_on_gpu:
            self.full_results[idx_start:idx_stop] = chunked_results
        else:
            chunked_results.get(out=self.full_results[idx_start:idx_stop])

    def initialize_full_results(self, chunked_results: Union[tuple, cp.ndarray]):
        output_array_size = (self.output_array_length, *chunked_results.shape[1:])
        if self.keep_on_gpu and self.pinned_results is not None:
            self.full_results = cp.empty(output_array_size, dtype=chunked_results.dtype)
        elif self.pinned_results is not None:
            self.full_results = self.pinned_results
        else:
            self.full_results = np.empty(output_array_size, dtype=chunked_results.dtype)


class Iterator:
    def __init__(self, inputs: InputArgumentsHandler, outputs: OutputResultsHandler, func: callable):
        self.inputs = inputs
        self.outputs = outputs
        self.func = func
        self.n_gpus = len(self.inputs.gpu_list)

    def run(self):
        gpu_idx = 0
        for iter in range(self.inputs.n_chunks):
            gpu = self.inputs.gpu_list[gpu_idx]
            cp.cuda.Device(gpu).use()

            self.inputs.update_chunked_list(iter, gpu_idx)
            chunked_results = self.func(*self.inputs.chunked_input_args)
            self.outputs.update_results(chunked_results, iter)

            gpu_idx = (iter + 1) % self.n_gpus


def get_chunk_indices(iter, chunk_size) -> tuple[int, int]:
    idx_start, idx_stop = iter * chunk_size, (iter + 1) * chunk_size
    return idx_start, idx_stop


def device_handling_wrapper(
    func: callable,
    options: DeviceOptions,
    chunkable_inputs_gpu_idx: List[int] = List[0],
    chunkable_inputs_cpu_idx: List[int] = [],
    common_inputs_gpu_idx: List[int] = [],
    pinned_results: Optional[np.ndarray] = None,
):
    @wraps(func)
    def wrapped(*args, **kwargs):
        ### case 1: cpu calculation ###
        if options.device_type == DeviceType.CPU:
            result = func(*args, **kwargs)
            return result

        inputs = InputArgumentsHandler(
            args,
            options.gpu_options,
            chunkable_inputs_gpu_idx,
            chunkable_inputs_cpu_idx,
            common_inputs_gpu_idx,
        )

        ### case 2: gpu calculation, 1 chunk ###
        if not options.gpu_options.chunking_enabled:
            inputs.move_inputs_to_gpu_in_one_chunk()
            result = func(*inputs.input_args, **kwargs)
            return result

        keep_on_gpu = all(
            [cp.get_array_module(input) is cp for input in inputs.chunkable_inputs_gpu]
        )

        outputs = OutputResultsHandler(
            options.gpu_options,
            inputs.chunkable_inputs_gpu[0].shape[0],
            keep_on_gpu,
            pinned_results,
        )

        # ### case 3: gpu calculation, multiple chunks ###
        iterator = Iterator(inputs, outputs, func)
        iterator.run()
        # for iter in range(inputs.n_chunks):
        #     # inputs.move_inputs_to_gpu()
        #     inputs.update_chunked_list(iter)
        #     chunked_results = func(*inputs.chunked_input_args, **kwargs)
        #     outputs.update_results(chunked_results, iter)

        return outputs.full_results

    return wrapped


# if __name__ == "__main__":
#     list_ref = [np.random.rand(10), np.random.rand(11), np.random.rand(12)]
#     chunkable_inputs_gpu_idx = [0, 1]
#     chunkable_inputs_cpu_idx = [2, 3, 4]
#     common_inputs_idx = []
#     inputs_list = InputArgumentsHandler(
#         list_ref, chunkable_inputs_gpu_idx, chunkable_inputs_cpu_idx, common_inputs_idx
#     )
#     inputs_list.chunkable_inputs_gpu
