import argparse
import h5py
import numpy as np
import cupy as cp
from llama.api.options.device import DeviceOptions, GPUOptions

from llama.api.options.projections import ProjectionOptions
from llama.api.options.task import AlignmentTaskOptions
from llama.gpu_utils import get_available_gpus, is_pinned, pin_memory
from llama.gpu_wrapper import device_handling_wrapper
from llama.data_structures.task import LaminographyAlignmentTask
from llama.data_structures.projections import ComplexProjections
from llama.api import enums

import llama.test_utils as tutils

n_arrays = 211
array_size = (23, 29)
a = np.random.rand(*array_size)  # 0
b = np.random.rand(n_arrays, *array_size)  # 1
c = np.random.rand(*array_size)  # 2
d = np.random.rand(n_arrays, *array_size)  # 3
e = np.random.rand(n_arrays, *array_size)  # 4
f = np.random.rand(*array_size)  # 5: no index specified
g = np.random.rand(n_arrays, *array_size)  # 6: cpu chunk index


def example_function(A, B, C, D, E, F, G, wrapped=True, return_tuple=False):
    if wrapped:
        assert type(F) is np.ndarray
        assert type(G) is np.ndarray
        result = A + B**2 + C**3 + D**4 + E**5 + cp.array(F) ** 6 + cp.array(G) ** 7
    else:
        result = A + B**2 + C**3 + D**4 + E**5 + F**6 + G**7
    if not return_tuple:
        return result
    else:
        return (result, result * 3, result * 5)


def initialize_input_positions_test(
    device_type=enums.DeviceType.GPU,
    chunk_length=13,
    n_gpus=None,
    chunking_enabled=True,
    return_tuple=False,
):
    gpu_indices = get_available_gpus()
    if n_gpus is None:
        n_gpus = len(gpu_indices)
    gpu_options = GPUOptions(
        chunking_enabled=chunking_enabled,
        chunk_length=chunk_length,
        n_gpus=n_gpus,
        gpu_indices=gpu_indices,
    )
    device_options = DeviceOptions(device_type=device_type, gpu=gpu_options)
    true_result = example_function(a, b, c, d, e, f, g, wrapped=False, return_tuple=return_tuple)
    return device_options, true_result


def test_gpu_wrapper_input_positions_1(pytestconfig=None):
    test_name = "test_gpu_wrapper_input_positions_1"

    device_options, true_result = initialize_input_positions_test()

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[1, 3, 4],
        common_inputs_for_gpu_idx=[0, 2],
        chunkable_inputs_for_cpu_idx=[6],
    )
    assert np.allclose(true_result, wrapped_function(a, b, c, d, e, f, g))
    tutils.print_passed_string(test_name)


def test_gpu_wrapper_input_positions_2(pytestconfig=None):
    test_name = "test_gpu_wrapper_input_positions_2"

    device_options, true_result = initialize_input_positions_test()

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[4, 1, 3],
        common_inputs_for_gpu_idx=[2, 0],
        chunkable_inputs_for_cpu_idx=[6],
    )
    assert np.allclose(true_result, wrapped_function(a, b, c, d, e, f, g))
    tutils.print_passed_string(test_name)


def test_gpu_wrapper_turned_off(pytestconfig=None):
    test_name = "test_gpu_wrapper_turned_off"

    device_options, true_result = initialize_input_positions_test(
        device_type=enums.DeviceType.CPU, n_gpus=1
    )

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[4, 1, 3],
        common_inputs_for_gpu_idx=[2, 0],
    )
    assert np.allclose(true_result, wrapped_function(a, b, c, d, e, f, g, False))
    tutils.print_passed_string(test_name)


def test_gpu_wrapper_single_chunk(pytestconfig=None):
    test_name = "test_gpu_wrapper_single_chunk"

    device_options, true_result = initialize_input_positions_test(
        device_type=enums.DeviceType.GPU, chunking_enabled=False
    )

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[4, 1, 3],
        common_inputs_for_gpu_idx=[2, 0],
        chunkable_inputs_for_cpu_idx=[6],
    )
    assert np.allclose(true_result, wrapped_function(a, b, c, d, e, f, g))
    tutils.print_passed_string(test_name)


def test_gpu_wrapper_pinned_outputs(pytestconfig=None):
    test_name = "test_gpu_wrapper_pinned_outputs"

    device_options, true_result = initialize_input_positions_test(
        device_type=enums.DeviceType.GPU,
    )

    pinned_results = pin_memory(np.empty_like(true_result))

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[4, 1, 3],
        common_inputs_for_gpu_idx=[2, 0],
        chunkable_inputs_for_cpu_idx=[6],
        pinned_results=pinned_results,
    )

    result = wrapped_function(a, b, c, d, e, f, g)
    assert result is pinned_results
    assert is_pinned(pinned_results)
    assert np.allclose(true_result, result)

    tutils.print_passed_string(test_name)


def test_gpu_wrapper_pinned_outputs_single_chunk(pytestconfig=None):
    test_name = "test_gpu_wrapper_pinned_outputs_single_chunk"

    device_options, true_result = initialize_input_positions_test(
        device_type=enums.DeviceType.GPU, chunking_enabled=False
    )

    pinned_results = pin_memory(np.empty_like(true_result))

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[4, 1, 3],
        common_inputs_for_gpu_idx=[2, 0],
        chunkable_inputs_for_cpu_idx=[6],
        pinned_results=pinned_results,
    )

    result = wrapped_function(a, b, c, d, e, f, g)
    assert result is pinned_results
    assert is_pinned(pinned_results)
    assert np.allclose(true_result, result)

    tutils.print_passed_string(test_name)


def test_gpu_wrapper_cupy_array_input(pytestconfig=None):
    test_name = "test_gpu_wrapper_cupy_array_input"

    device_options, true_result = initialize_input_positions_test(
        device_type=enums.DeviceType.GPU,
    )

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[4, 1, 3],
        common_inputs_for_gpu_idx=[2, 0],
        chunkable_inputs_for_cpu_idx=[6],
    )

    result = wrapped_function(cp.array(a), cp.array(b), cp.array(c), cp.array(d), cp.array(e), f, g)
    assert type(result) is cp.ndarray
    assert np.allclose(true_result, result.get())

    tutils.print_passed_string(test_name)


def test_gpu_wrapper_cupy_array_input_pinned_output(pytestconfig=None):
    test_name = "test_gpu_wrapper_cupy_array_input_pinned_output"

    device_options, true_result = initialize_input_positions_test(
        device_type=enums.DeviceType.GPU,
    )

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[4, 1, 3],
        common_inputs_for_gpu_idx=[2, 0],
        chunkable_inputs_for_cpu_idx=[6],
        pinned_results=cp.empty_like(true_result),
    )

    result = wrapped_function(cp.array(a), cp.array(b), cp.array(c), cp.array(d), cp.array(e), f, g)
    assert type(result) is cp.ndarray
    assert np.allclose(true_result, result.get())

    tutils.print_passed_string(test_name)


def test_gpu_wrapper_fewer_chunks_than_gpus(pytestconfig=None):
    test_name = "test_gpu_wrapper_fewer_chunks_than_gpus"

    device_options, true_result = initialize_input_positions_test(
        device_type=enums.DeviceType.GPU, chunk_length=100
    )

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[4, 1, 3],
        common_inputs_for_gpu_idx=[2, 0],
        chunkable_inputs_for_cpu_idx=[6],
    )

    result = wrapped_function(a, b, c, d, e, f, g)
    assert np.allclose(true_result, result)

    tutils.print_passed_string(test_name)


def test_gpu_wrapper_with_tuple_output(pytestconfig=None):
    test_name = "test_gpu_wrapper_with_tuple_output"

    device_options, true_result = initialize_input_positions_test(
        device_type=enums.DeviceType.GPU, return_tuple=True
    )

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[4, 1, 3],
        common_inputs_for_gpu_idx=[2, 0],
        chunkable_inputs_for_cpu_idx=[6],
    )

    result = wrapped_function(a, b, c, d, e, f, g, return_tuple=True)
    for i in range(len(true_result)):
        assert np.allclose(true_result[i], result[i])

    tutils.print_passed_string(test_name)


def test_gpu_wrapper_with_tuple_output_pinned(pytestconfig=None):
    test_name = "test_gpu_wrapper_with_tuple_output_pinned"

    device_options, true_result = initialize_input_positions_test(
        device_type=enums.DeviceType.GPU, return_tuple=True
    )

    pinned_results = tuple([pin_memory(np.empty_like(result)) for result in true_result])

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[4, 1, 3],
        common_inputs_for_gpu_idx=[2, 0],
        chunkable_inputs_for_cpu_idx=[6],
        pinned_results=pinned_results,
    )

    result = wrapped_function(a, b, c, d, e, f, g, return_tuple=True)
    for i in range(len(true_result)):
        assert is_pinned(pinned_results[i])
        assert pinned_results[i] is result[i]
        assert np.allclose(true_result[i], result[i])

    tutils.print_passed_string(test_name)


def test_gpu_wrapper_cupy_array_input_pinned_output(pytestconfig=None):
    test_name = "test_gpu_wrapper_cupy_array_input_pinned_output"

    device_options, true_result = initialize_input_positions_test(
        device_type=enums.DeviceType.GPU,
    )

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[4, 1, 3],
        common_inputs_for_gpu_idx=[2, 0],
        chunkable_inputs_for_cpu_idx=[6],
        pinned_results=pin_memory(np.empty_like(true_result)),
    )

    result = wrapped_function(cp.array(a), cp.array(b), cp.array(c), cp.array(d), cp.array(e), f, g)
    assert type(result) is np.ndarray
    assert np.allclose(true_result, result)

    tutils.print_passed_string(test_name)


if __name__ == "__main__":
    test_gpu_wrapper_input_positions_1()
    test_gpu_wrapper_input_positions_2()
    test_gpu_wrapper_turned_off()
    test_gpu_wrapper_single_chunk()
    test_gpu_wrapper_pinned_outputs()
    test_gpu_wrapper_pinned_outputs_single_chunk()
    test_gpu_wrapper_cupy_array_input()
    test_gpu_wrapper_cupy_array_input_pinned_output()
    test_gpu_wrapper_fewer_chunks_than_gpus()
    test_gpu_wrapper_with_tuple_output()
    test_gpu_wrapper_with_tuple_output_pinned()
    test_gpu_wrapper_cupy_array_input_pinned_output()
