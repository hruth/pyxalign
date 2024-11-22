import argparse
import h5py
import numpy as np
from llama.api.options.device import DeviceOptions, GPUOptions

from llama.api.options.projections import ProjectionOptions
from llama.api.options.task import AlignmentTaskOptions
from llama.gpu_utils import get_available_gpus, is_pinned, pin_memory
from llama.gpu_wrapper import device_handling_wrapper
from llama.task import LaminographyAlignmentTask
from llama.projections import ComplexProjections
from llama.api import enums

import llama.test_utils as tutils

n_arrays = 211
array_size = (23, 29)
a = np.random.rand(*array_size)
b = np.random.rand(n_arrays, *array_size)
c = np.random.rand(*array_size)
d = np.random.rand(n_arrays, *array_size)
e = np.random.rand(n_arrays, *array_size)


def example_function(A, B, C, D, E):
    return A + B**2 + C**3 + D**4 + E**5


def initialize_input_positions_test(
    device_type=enums.DeviceType.GPU, chunk_size=13, n_gpus=None, chunking_enabled=True
):
    gpu_indices = get_available_gpus()
    if n_gpus is None:
        n_gpus = len(gpu_indices)
    gpu_options = GPUOptions(
        chunking_enabled=chunking_enabled,
        chunk_size=chunk_size,
        n_gpus=n_gpus,
        gpu_indices=gpu_indices,
    )
    device_options = DeviceOptions(device_type=device_type, gpu_options=gpu_options)
    true_result = example_function(a, b, c, d, e)
    return device_options, true_result


def test_gpu_wrapper_input_positions_1(pytestconfig):
    test_name = "test_gpu_wrapper_input_positions_1"

    device_options, true_result = initialize_input_positions_test()

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[1, 3, 4],
        common_inputs_for_gpu_idx=[0, 2],
    )
    assert np.allclose(true_result, wrapped_function(a, b, c, d, e))
    tutils.print_passed_string(test_name)


def test_gpu_wrapper_input_positions_2(pytestconfig):
    test_name = "test_gpu_wrapper_input_positions_2"

    device_options, true_result = initialize_input_positions_test()

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[4, 1, 3],
        common_inputs_for_gpu_idx=[2, 0],
    )
    assert np.allclose(true_result, wrapped_function(a, b, c, d, e))
    tutils.print_passed_string(test_name)


def test_gpu_wrapper_turned_off(pytestconfig):
    test_name = "test_gpu_wrapper_turned_off"

    device_options, true_result = initialize_input_positions_test(
        device_type=enums.DeviceType.CPU,
        n_gpus=1
    )

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
    )
    assert np.allclose(true_result, wrapped_function(a, b, c, d, e))
    tutils.print_passed_string(test_name)

def test_gpu_wrapper_single_chunk(pytestconfig):
    test_name = "test_gpu_wrapper_single_chunk"

    device_options, true_result = initialize_input_positions_test(
        device_type=enums.DeviceType.GPU,
        chunking_enabled=False
    )

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
    )
    assert np.allclose(true_result, wrapped_function(a, b, c, d, e))
    tutils.print_passed_string(test_name)


def test_gpu_wrapper_pinned_outputs(pytestconfig):
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
        pinned_results=pinned_results
    )
    
    result = wrapped_function(a, b, c, d, e)
    assert result is pinned_results
    assert is_pinned(pinned_results)
    assert np.allclose(true_result, result)

    tutils.print_passed_string(test_name)

def test_gpu_wrapper_pinned_outputs_single_chunk(pytestconfig):
    test_name = "test_gpu_wrapper_pinned_outputs_single_chunk"

    device_options, true_result = initialize_input_positions_test(
        device_type=enums.DeviceType.GPU,
        chunking_enabled=False
    )

    pinned_results = pin_memory(np.empty_like(true_result))

    wrapped_function = device_handling_wrapper(
        func=example_function,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[4, 1, 3],
        common_inputs_for_gpu_idx=[2, 0],
        pinned_results=pinned_results
    )
    
    result = wrapped_function(a, b, c, d, e)
    assert result is pinned_results
    assert is_pinned(pinned_results)
    assert np.allclose(true_result, result)

    tutils.print_passed_string(test_name)


if __name__ == "__main__":
    # test_gpu_wrapper_input_positions_1(None)
    # test_gpu_wrapper_input_positions_2(None)
    # test_gpu_wrapper_turned_off(None)
    test_gpu_wrapper_single_chunk(None)
    # test_gpu_wrapper_pinned_outputs(None)
    # test_gpu_wrapper_pinned_outputs_single_chunk(None)
