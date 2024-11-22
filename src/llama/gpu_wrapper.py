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
        chunkable_inputs_for_gpu_idx: List[int],
        chunkable_inputs_for_cpu_idx: List[int],
        common_inputs_for_gpu_idx: List[int],
    ):
        self.args = list(args)
        # self.input_args = args
        self.chunkable_inputs_for_gpu_idx = chunkable_inputs_for_gpu_idx
        self.chunkable_inputs_for_cpu_idx = chunkable_inputs_for_cpu_idx
        self.common_inputs_for_gpu_idx = common_inputs_for_gpu_idx
        self.options = options
        self.n_chunks = int(np.ceil(len(self.chunkable_inputs_for_gpu[0]) / options.chunk_size))
        self.initialize_devices()
        self.initialize_single_iter_args()
        self.initialize_chunked_args()
        self.move_common_inputs_to_gpu()

    @property
    def chunkable_inputs_for_gpu(self) -> List[ArrayType]:
        return [self.args[i] for i in self.chunkable_inputs_for_gpu_idx]

    def set_chunkable_inputs_for_gpu(self, new_value: ArrayType, idx: int):
        self.args[self.chunkable_inputs_for_gpu_idx[idx]] = new_value

    def initialize_devices(self):
        gpu_utils.check_gpu_list(self.options.n_gpus, self.options.gpu_indices)
        self.gpu_list = self.options.gpu_indices[: self.options.n_gpus]

    def move_inputs_to_gpu_in_one_chunk(self):
        # single gpu, single chunk
        i = 0
        for input in self.chunkable_inputs_for_gpu:
            self.set_chunkable_inputs_for_gpu(cp.array(input), i)
            i += 1

    def initialize_single_iter_args(self):
        """Initialize the list of args that will be passed to the wrapped
        function on each iteration"""
        self.current_iter_args = list(self.args)

    def initialize_chunked_args(self):
        "Initialize the class that handles moving chunks from cpu to gpu"
        self.gpu_stager = GPUStager(
            self.gpu_list,
            self.options.chunk_size,
            self.n_chunks,
            self.chunkable_inputs_for_gpu,
        )

    def move_common_inputs_to_gpu(self):
        "Move all common inputs (inputs that are passed in full) onto the gpu(s)"
        self.common_inputs_on_gpu = []
        for arg_idx in self.common_inputs_for_gpu_idx:
            for gpu in self.gpu_list:
                with cp.cuda.Device(gpu).use():
                    self.common_inputs_on_gpu += [[cp.array(self.args[arg_idx])]]

    def update_chunked_list(self, iter: int, gpu_idx: int):
        "Update the arguments that will be passed to the wrapped function"
        self.update_gpu_chunks(iter, gpu_idx)
        self.update_cpu_chunks(iter)
        self.update_common_inputs(gpu_idx)

    def update_gpu_chunks(self, iter: int, gpu_idx: int):
        """Update the args for this iteration with the inputs that are chunked
        and put on the gpu"""
        self.gpu_stager.update_chunked_inputs(gpu_idx, iter)
        for i in range(len(self.chunkable_inputs_for_gpu_idx)):
            arg_idx = self.chunkable_inputs_for_gpu_idx[i]
            self.current_iter_args[arg_idx] = self.gpu_stager.get_next_chunked_inputs(gpu_idx, i)

    def update_cpu_chunks(self, iter: int):
        """Update the args for this iteration with the inputs that are chunked
        and kept on the cpu"""
        idx_start, idx_stop = get_chunk_indices(iter, self.options.chunk_size)
        for arg_idx in self.chunkable_inputs_for_cpu_idx:
            self.current_iter_args[arg_idx] = self.args[arg_idx][idx_start:idx_stop]

    def update_common_inputs(self, gpu_idx: int):
        """Update the args for this iteration with the inputs that are put on
        the gpu and not chunked"""
        for i in range(len(self.common_inputs_for_gpu_idx)):
            arg_idx = self.common_inputs_for_gpu_idx[i]
            self.current_iter_args[arg_idx] = self.common_inputs_on_gpu[gpu_idx][i]


class GPUStager:
    """
    Creates and updates `list_of_arrays`, a list of GPU arrays that
    will hold chunks of the input arrays that need to be transferred
    to the GPU. `list_of_arrays` essentially serves as a "staging area"
    for the cupy arrays before they are passed to the wrapped function.

    The attribute `list_of_arrays` has `n_gpu` elements. Each of those
    elements is a list of of arrays with length `chunk_size`.
    """

    def __init__(
        self,
        gpu_list: List[int],
        chunk_size: int,
        n_chunks: int,
        chunkable_inputs_for_gpu: List[ArrayType],
    ):
        self.gpu_list = gpu_list
        self.chunkable_inputs_for_gpu = chunkable_inputs_for_gpu
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.initialize_list_of_arrays()

    @property
    def n_gpus(self):
        return len(self.gpu_list)

    def initialize_list_of_arrays(self):
        self.list_of_arrays: List[List[cp.ndarray]] = []
        for gpu in self.gpu_list:
            with cp.cuda.Device(gpu).use():
                self.list_of_arrays += [self.create_gpu_containers()]

    def create_gpu_containers(self) -> List[cp.ndarray]:
        "Create the list of cupy arrays that will hold chunks of the chunkable inputs for gpu"
        return [self.create_container_element(array) for array in self.chunkable_inputs_for_gpu]

    def create_container_element(self, array: ArrayType) -> cp.ndarray:
        "Create the cupy array that will hold chunks from the input numpy array"
        array_size = (self.chunk_size, *array.shape[1:])
        return cp.empty(array_size, dtype=array.dtype)

    def update_chunked_inputs(self, gpu_idx: int, iter: int):
        "Insert data from each full array into each cupy chunk array"
        for i in range(len(self.chunkable_inputs_for_gpu)):
            self.insert_next_chunk_into_cupy_array(gpu_idx, iter, i)

    def insert_next_chunk_into_cupy_array(self, gpu_idx: int, iter: int, arg_idx: int):
        """Update the cupy array with the next chunk that needs to be passed
        to the wrapped function"""
        idx_start, idx_stop = get_chunk_indices(iter, self.chunk_size)
        numpy_chunk_array = self.chunkable_inputs_for_gpu[arg_idx][idx_start:idx_stop]
        self.revised_chunk_length = len(numpy_chunk_array)
        cupy_chunk_array = self.list_of_arrays[gpu_idx][arg_idx][: self.revised_chunk_length]
        cupy_chunk_array.set(numpy_chunk_array)

    def get_next_chunked_inputs(self, gpu_idx: int, list_idx: int) -> cp.ndarray:
        return self.list_of_arrays[gpu_idx][list_idx][: self.revised_chunk_length]


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
    def __init__(
        self,
        inputs: InputArgumentsHandler,
        outputs: OutputResultsHandler,
        func: callable,
    ):
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
            chunked_results = self.func(*self.inputs.current_iter_args)
            self.outputs.update_results(chunked_results, iter)

            gpu_idx = (iter + 1) % self.n_gpus


def get_chunk_indices(iter, chunk_size) -> tuple[int, int]:
    idx_start, idx_stop = iter * chunk_size, (iter + 1) * chunk_size
    return idx_start, idx_stop


def device_handling_wrapper(
    func: callable,
    options: DeviceOptions,
    chunkable_inputs_for_gpu_idx: List[int] = List[0],
    chunkable_inputs_for_cpu_idx: List[int] = [],
    common_inputs_for_gpu_idx: List[int] = [],
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
            chunkable_inputs_for_gpu_idx,
            chunkable_inputs_for_cpu_idx,
            common_inputs_for_gpu_idx,
        )

        ### case 2: gpu calculation, 1 chunk ###
        if not options.gpu_options.chunking_enabled:
            inputs.move_inputs_to_gpu_in_one_chunk()
            result = func(*inputs.args, **kwargs)
            return result

        keep_on_gpu = all(
            [cp.get_array_module(input) is cp for input in inputs.chunkable_inputs_for_gpu]
        )

        outputs = OutputResultsHandler(
            options.gpu_options,
            inputs.chunkable_inputs_for_gpu[0].shape[0],
            keep_on_gpu,
            pinned_results,
        )

        # ### case 3: gpu calculation, multiple chunks and potentially multiple GPUs ###
        iterator = Iterator(inputs, outputs, func)
        iterator.run()

        return outputs.full_results

    return wrapped
