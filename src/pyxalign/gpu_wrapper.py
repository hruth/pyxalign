from functools import wraps
from typing import Callable, List, Optional, TypeVar, Union, Any
import cupy as cp
import numpy as np
from tqdm import tqdm

from pyxalign import gpu_utils

from pyxalign.api.enums import DeviceType
from pyxalign.api.options.device import DeviceOptions, GPUOptions

from pyxalign.api.types import ArrayType
from pyxalign.timing.timer_utils import timer

# To do:
# - check gpu profiling results with nsight-sys
# - allow chunking on cpu
#   - in other code, you should update the pinned_results=None
#     that are automatically used for cpu memory config to
#     be an unpinned numpy array. I wonder if there is any
#     utility to pinning memory for this case.
# - Later, you could update the gpu wrapper
#   to move arrays to the gpu in chunks if they
#   are on the cpu. For now, this is handled externally.

T = TypeVar("T", bound=Callable[..., Any])

timer_enabled = False
"This timer should only ever be turned on for debugging purposes."


class InputArgumentsHandler:
    @timer(enabled=timer_enabled)
    def __init__(
        self,
        args: tuple,
        options: GPUOptions,
        chunkable_inputs_for_gpu_idx: List[int],
        chunkable_inputs_for_cpu_idx: List[int],
        common_inputs_for_gpu_idx: List[int],
        inputs_already_on_gpu: bool,
        stream_list: List[cp.cuda.stream.Stream],
    ):
        self.args = list(args)
        self.options = options
        self.chunkable_inputs_for_gpu_idx = chunkable_inputs_for_gpu_idx
        self.chunkable_inputs_for_cpu_idx = chunkable_inputs_for_cpu_idx
        self.common_inputs_for_gpu_idx = common_inputs_for_gpu_idx
        self.inputs_already_on_gpu = inputs_already_on_gpu
        self.stream_list = stream_list
        if self.options.chunking_enabled:
            self.chunk_length = options.chunk_length
            self.n_chunks = int(np.ceil(len(self.chunkable_inputs_for_gpu[0]) / self.chunk_length))
        else:
            self.n_chunks = 1
            self.chunk_length = len(self.chunkable_inputs_for_gpu[0])
        self.initialize_devices()
        self.initialize_single_iter_args()
        self.initialize_chunked_args()
        self.move_common_inputs_to_gpu()

    @property
    def chunkable_inputs_for_gpu(self) -> List[ArrayType]:
        return [self.args[i] for i in self.chunkable_inputs_for_gpu_idx]

    @timer(enabled=timer_enabled)
    def set_chunkable_inputs_for_gpu(self, new_value: ArrayType, idx: int):
        self.args[self.chunkable_inputs_for_gpu_idx[idx]] = new_value

    @timer(enabled=timer_enabled)
    def initialize_devices(self):
        if self.inputs_already_on_gpu:
            self.gpu_list = (self.chunkable_inputs_for_gpu[0].device,)
        else:
            gpu_utils.check_gpu_list(self.options.n_gpus, self.options.gpu_indices)
            self.gpu_list = self.options.gpu_indices[: self.options.n_gpus]
            # The number of gpus should not exceed the number of chunks
            self.gpu_list = self.gpu_list[: self.n_chunks]

    @timer(enabled=timer_enabled)
    def initialize_single_iter_args(self):
        """Initialize the list of args that will be passed to the wrapped
        function on each iteration"""
        self.current_iter_args = list(self.args)

    @timer(enabled=timer_enabled)
    def initialize_chunked_args(self):
        "Initialize the class that handles moving chunks from cpu to gpu"
        self.gpu_stager = GPUStager(
            self.gpu_list,
            self.chunk_length,
            self.n_chunks,
            self.chunkable_inputs_for_gpu,
            self.inputs_already_on_gpu,
            self.stream_list,
        )

    @timer(enabled=timer_enabled)
    def move_common_inputs_to_gpu(self):
        "Move all common inputs (inputs that are passed in full) onto the gpu(s)"
        self.common_inputs_on_gpu = []
        for gpu_idx, gpu in enumerate(self.gpu_list):
            with cp.cuda.Device(gpu).use():
                with self.stream_list[gpu_idx]:
                    self.common_inputs_on_gpu += [
                        [cp.array(self.args[arg_idx]) for arg_idx in self.common_inputs_for_gpu_idx]
                    ]

    @timer(enabled=timer_enabled)
    def update_chunked_list(self, iter: int, gpu_idx: int):
        "Update the arguments that will be passed to the wrapped function"
        self.update_gpu_chunks(iter, gpu_idx)
        self.update_cpu_chunks(iter)
        self.update_common_inputs(gpu_idx)

    @timer(enabled=timer_enabled)
    def update_gpu_chunks(self, iter: int, gpu_idx: int):
        """Update the args for this iteration with the inputs that are chunked
        and put on the gpu"""
        self.gpu_stager.update_chunked_inputs(gpu_idx, iter)
        for i in range(len(self.chunkable_inputs_for_gpu_idx)):
            arg_idx = self.chunkable_inputs_for_gpu_idx[i]
            self.current_iter_args[arg_idx] = self.gpu_stager.get_next_chunked_inputs(
                gpu_idx, i, iter
            )

    @timer(enabled=timer_enabled)
    def update_cpu_chunks(self, iter: int):
        """Update the args for this iteration with the inputs that are chunked
        and kept on the cpu"""
        idx_start, idx_stop = get_chunk_indices(iter, self.chunk_length)
        for arg_idx in self.chunkable_inputs_for_cpu_idx:
            self.current_iter_args[arg_idx] = self.args[arg_idx][idx_start:idx_stop]

    @timer(enabled=timer_enabled)
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
    elements is a list of of arrays with length `chunk_length`.
    """

    @timer(enabled=timer_enabled)
    def __init__(
        self,
        gpu_list: List[int],
        chunk_length: int,
        n_chunks: int,
        chunkable_inputs_for_gpu: List[ArrayType],
        inputs_already_on_gpu: bool,
        stream_list: List[cp.cuda.stream.Stream],
    ):
        self.gpu_list = gpu_list
        self.chunkable_inputs_for_gpu = chunkable_inputs_for_gpu
        self.chunk_length = chunk_length
        self.n_chunks = n_chunks
        self.inputs_already_on_gpu = inputs_already_on_gpu
        self.stream_list = stream_list
        self.initialize_list_of_arrays()

    @property
    def n_gpus(self):
        return len(self.gpu_list)

    @timer(enabled=timer_enabled)
    def initialize_list_of_arrays(self):
        self.list_of_arrays: List[List[cp.ndarray]] = []
        for gpu_idx, gpu in enumerate(self.gpu_list):
            with cp.cuda.Device(gpu).use():
                with self.stream_list[gpu_idx]:
                    self.list_of_arrays += [self.create_gpu_containers()]

    @timer(enabled=timer_enabled)
    def create_gpu_containers(self) -> List[cp.ndarray]:
        "Create the list of cupy arrays that will hold chunks of the chunkable inputs for gpu"
        return [self.create_container_element(array) for array in self.chunkable_inputs_for_gpu]

    @timer(enabled=timer_enabled)
    def create_container_element(self, array: ArrayType) -> cp.ndarray:
        "Create the cupy array that will hold chunks from the input numpy array"
        array_size = (self.chunk_length, *array.shape[1:])
        return cp.empty(array_size, dtype=array.dtype)

    @timer(enabled=timer_enabled)
    def update_chunked_inputs(self, gpu_idx: int, iter: int):
        "Insert data from each full array into each cupy chunk array"
        if self.inputs_already_on_gpu:
            return
        for i in range(len(self.chunkable_inputs_for_gpu)):
            self.insert_next_chunk_into_cupy_array(gpu_idx, iter, i)

    @timer(enabled=timer_enabled)
    def insert_next_chunk_into_cupy_array(self, gpu_idx: int, iter: int, arg_idx: int):
        """Update the cupy array with the next chunk that needs to be passed
        to the wrapped function"""
        idx_start, idx_stop = get_chunk_indices(iter, self.chunk_length)
        numpy_chunk_array = self.chunkable_inputs_for_gpu[arg_idx][idx_start:idx_stop]
        self.revised_chunk_length = len(numpy_chunk_array)
        cupy_chunk_array = self.list_of_arrays[gpu_idx][arg_idx][: self.revised_chunk_length]
        if type(numpy_chunk_array) is np.ndarray:
            cupy_chunk_array.set(numpy_chunk_array)
            # self.list_of_arrays[gpu_idx][arg_idx][: self.revised_chunk_length].set(
            #     self.chunkable_inputs_for_gpu[arg_idx][idx_start:idx_stop]
            # )
        elif type(numpy_chunk_array) is cp.ndarray:
            cupy_chunk_array[:] = numpy_chunk_array
            # self.list_of_arrays[gpu_idx][arg_idx][: self.revised_chunk_length][:] = (
            #     self.chunkable_inputs_for_gpu[arg_idx][idx_start:idx_stop]
            # )

        # idx_start, idx_stop = get_chunk_indices(iter, self.chunk_length)
        # self.revised_chunk_length = len(self.chunkable_inputs_for_gpu[arg_idx])
        # cupy_chunk_array = self.list_of_arrays[gpu_idx][arg_idx]
        # if type(self.chunkable_inputs_for_gpu[arg_idx]) is np.ndarray:
        #     inline_timer = InlineTimer("set_data")
        #     inline_timer.start()
        #     cupy_chunk_array.set(self.chunkable_inputs_for_gpu[arg_idx][idx_start:idx_stop])
        #     inline_timer.end()
        # elif type(self.chunkable_inputs_for_gpu[arg_idx]) is cp.ndarray:
        #     cupy_chunk_array[:] = self.chunkable_inputs_for_gpu[arg_idx][idx_start:idx_stop]

    @timer(enabled=timer_enabled)
    def get_next_chunked_inputs(self, gpu_idx: int, list_idx: int, iter: int) -> cp.ndarray:
        if self.inputs_already_on_gpu:
            idx_start, idx_stop = get_chunk_indices(iter, self.chunk_length)
            return self.chunkable_inputs_for_gpu[list_idx][idx_start:idx_stop]
        else:
            return self.list_of_arrays[gpu_idx][list_idx][: self.revised_chunk_length]


class OutputResultsHandler:
    @timer(enabled=timer_enabled)
    def __init__(
        self,
        chunk_length: int,
        output_array_length: int,
        already_on_gpu: bool,
        pinned_results: Optional[np.ndarray] = None,
    ):
        self.chunk_length = chunk_length
        self.output_array_length = output_array_length
        self.already_on_gpu = already_on_gpu
        self.pinned_results = pinned_results

    @timer(enabled=timer_enabled)
    def update_results(self, chunked_results: Union[tuple, cp.ndarray], iter: int):
        # need to update to work with tuples later
        # if not isinstance(chunked_results, tuple):
        # chunked_results = (chunked_results,)
        if iter == 0:
            self.initialize_full_results(chunked_results)
        self.insert_into_full_results(chunked_results, iter)

    @timer(enabled=timer_enabled)
    def insert_into_full_results(self, chunked_results: Union[tuple, cp.ndarray], iter: int):
        idx_start, idx_stop = get_chunk_indices(iter, self.chunk_length)
        if type(chunked_results) is not tuple:
            chunked_results = (chunked_results,)
        # iterate through the tuple of results
        for i in range(len(chunked_results)):
            if self.already_on_gpu and type(self.full_results[i]) is not np.ndarray:
                self.full_results[i][idx_start:idx_stop] = chunked_results[i]
            else:
                chunked_results[i].get(out=self.full_results[i][idx_start:idx_stop])

    @timer(enabled=timer_enabled)
    def initialize_full_results(self, chunked_results: Union[tuple, cp.ndarray]):
        "Create a tuple that will hold the outputs of the wrapped function"
        if self.pinned_results is not None:
            if type(self.pinned_results) is not tuple:
                self.full_results = (self.pinned_results,)
            else:
                self.full_results = self.pinned_results
            return

        if type(chunked_results) is not tuple:
            chunked_results = (chunked_results,)

        self.full_results: tuple = ()

        for chunked_result in chunked_results:
            output_array_size = (self.output_array_length, *chunked_result.shape[1:])
            if self.already_on_gpu:
                self.full_results += (cp.empty(output_array_size, dtype=chunked_result.dtype),)
            else:
                self.full_results += (np.empty(output_array_size, dtype=chunked_result.dtype),)


class Iterator:
    @timer(enabled=timer_enabled)
    def __init__(
        self,
        inputs: InputArgumentsHandler,
        outputs: OutputResultsHandler,
        func: T,
        display_progress_bar: bool = False,
        stream_list: List[cp.cuda.stream.Stream] = None,
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.func = func
        self.n_gpus = len(self.inputs.gpu_list)
        self.display_progress_bar = display_progress_bar
        self.stream_list = stream_list

    @timer(enabled=timer_enabled)
    def run(self, kwargs: dict = {}):
        gpu_idx = 0
        if self.display_progress_bar:
            iterate_over = tqdm(range(self.inputs.n_chunks), desc=self.func.__name__)
        else:
            iterate_over = range(self.inputs.n_chunks)
        for iter in iterate_over:
            gpu = self.inputs.gpu_list[gpu_idx]
            cp.cuda.Device(gpu).use()
            with self.stream_list[gpu_idx]:
                self.inputs.update_chunked_list(iter, gpu_idx)
                chunked_results = self.func(*self.inputs.current_iter_args, **kwargs)
                self.outputs.update_results(chunked_results, iter)
            gpu_idx = (iter + 1) % self.n_gpus
        if len(self.outputs.full_results) == 1:
            self.outputs.full_results = self.outputs.full_results[0]


@timer(enabled=timer_enabled)
def get_chunk_indices(iter, chunk_length) -> tuple[int, int]:
    idx_start, idx_stop = iter * chunk_length, (iter + 1) * chunk_length
    return idx_start, idx_stop


@timer(enabled=timer_enabled)
def check_if_arrays_are_on_same_device(
    args: tuple, chunkable_inputs_for_gpu_idx: List[int]
) -> bool:
    idx = chunkable_inputs_for_gpu_idx
    all_arrays_on_cpu = all([cp.get_array_module(args[i]) is np for i in idx])
    all_arrays_on_gpu = all([cp.get_array_module(args[i]) is cp for i in idx])
    if all_arrays_on_gpu:
        all_arrays_on_same_gpu = all([args[i].device == args[0].device for i in idx])
    else:
        all_arrays_on_same_gpu = False
    all_arrays_on_same_device = all_arrays_on_cpu or (all_arrays_on_gpu and all_arrays_on_same_gpu)
    if not all_arrays_on_same_device:
        raise Exception("chunkable_inputs_for_gpu must be on the same device!")

    return all_arrays_on_same_gpu


@timer(enabled=timer_enabled)
@gpu_utils.memory_releasing_error_handler
def device_handling_wrapper(
    func: T,
    options: DeviceOptions,
    chunkable_inputs_for_gpu_idx: List[int] = [0],
    chunkable_inputs_for_cpu_idx: List[int] = [],
    common_inputs_for_gpu_idx: List[int] = [],
    pinned_results: Optional[Union[np.ndarray, tuple]] = None,
    display_progress_bar: bool = False,
) -> T:
    """Wrapper that efficiently splits inputs into chunks and transfers them between
    the gpu and cpu.

    When using the wrapped function, the input arguments referred to by
    `chunkable_inputs_for_gpu_idx`, `chunkable_inputs_for_cpu_idx`, and
    `common_inputs_for_gpu_idx` must be passed in as args, not kwargs.
    """

    @wraps(func)
    @timer(enabled=timer_enabled)
    def wrapped(*args, **kwargs):
        # Check inputs
        inputs_are_cupy_arrays = check_if_arrays_are_on_same_device(
            args, chunkable_inputs_for_gpu_idx
        )

        ### case 1: cpu calculation ###
        if options.device_type == DeviceType.CPU:
            if display_progress_bar:
                for i in tqdm(range(1), desc=func.__name__):
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result

        ### case 2: on gpu, multiple chunks, and potentially multiple GPUs ###

        # Synchronize with null streams before and after execution
        synchronize_with_null(options)

        stream_list = create_streams(options)

        inputs = InputArgumentsHandler(
            args,
            options.gpu,
            chunkable_inputs_for_gpu_idx,
            chunkable_inputs_for_cpu_idx,
            common_inputs_for_gpu_idx,
            inputs_are_cupy_arrays,
            stream_list,
        )

        outputs = OutputResultsHandler(
            inputs.chunk_length,
            inputs.chunkable_inputs_for_gpu[0].shape[0],
            inputs_are_cupy_arrays,
            pinned_results,
        )

        iterator = Iterator(inputs, outputs, func, display_progress_bar, stream_list)
        iterator.run(kwargs)

        synchronize_with_null(options)

        return outputs.full_results

    return wrapped


@timer()
def synchronize_with_null(device_options: DeviceOptions):
    gpu_list = device_options.gpu.gpu_indices[: device_options.gpu.n_gpus]
    for gpu in gpu_list:
        with cp.cuda.Device(gpu):
            cp.cuda.stream.Stream(null=True).synchronize()


@timer()
def create_streams(device_options: DeviceOptions):
    gpu_list = device_options.gpu.gpu_indices[: device_options.gpu.n_gpus]
    stream_list = []
    for gpu in gpu_list:
        with cp.cuda.Device(gpu):
            stream_list += [cp.cuda.stream.Stream()]
    return stream_list
