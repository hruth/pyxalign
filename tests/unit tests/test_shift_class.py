import argparse
import time
import numpy as np
import cupy as cp
import pytest
from llama.gpu_utils import get_available_gpus, is_pinned
from llama.data_structures.projections import ComplexProjections
from llama.api.options.transform import ShiftOptions
from llama.transformations.classes import Shifter
from llama.api import enums

import llama.test_utils as tutils
from llama.api.types import r_type

filename = "cSAXS_projections_downsampling16.h5"
repeat_array = False
n_reps = 4


def shift_projections(
    complex_projections: ComplexProjections,
    device_type: enums.DeviceType,
    shift_type: enums.ShiftType,
    chunking_enabled: bool = False,
    n_gpus: int = 1,
    gpu_indices: tuple = (0,),
    chunk_length: int = 100,
):

    n_images = len(complex_projections.angles)
    shift_array_size = (complex_projections.n_projections, 2)
    shift = (
        np.zeros(shift_array_size)
        + np.linspace(0, complex_projections.data.shape[2], n_images)[:, None]
    ).astype(r_type)

    shift_options = ShiftOptions()
    shift_options.enabled = True
    shift_options.device.device_type = device_type
    shift_options.type = shift_type
    shift_options.device.gpu.chunk_length = chunk_length
    shift_options.device.gpu.chunking_enabled = chunking_enabled
    shift_options.device.gpu.n_gpus = n_gpus
    shift_options.device.gpu.gpu_indices = gpu_indices
    shifter = Shifter(shift_options)
    shifted_projections = shifter.run(
        complex_projections.data, shift, pinned_results=complex_projections.data
    )

    return shifted_projections


def test_fft_shift_class_cpu(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_fft_shift_class_cpu"
    complex_projections = tutils.prepare_data(filename)
    if repeat_array:
        tutils.repeat_array(complex_projections, n_reps)
    shifted_projections = shift_projections(
        complex_projections, enums.DeviceType.CPU, enums.ShiftType.FFT
    )
    tutils.check_or_record_results(
        shifted_projections,
        test_name,
        test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results
    )


def test_fft_shift_class_gpu(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_fft_shift_class_gpu"
    comparison_test_name = "test_fft_shift_class_cpu"
    complex_projections = tutils.prepare_data(filename)
    if repeat_array:
        tutils.repeat_array(complex_projections, n_reps)
    t0 = time.time()
    shifted_projections = shift_projections(
        complex_projections, enums.DeviceType.GPU, enums.ShiftType.FFT
    )
    print(test_name, time.time() - t0)
    tutils.check_or_record_results(
        shifted_projections,
        test_name,
        comparison_test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results
    )


def test_circ_shift_class_gpu(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_circ_shift_class_cpu"
    complex_projections = tutils.prepare_data(filename)
    if repeat_array:
        tutils.repeat_array(complex_projections, n_reps)
    shifted_projections = shift_projections(
        complex_projections, enums.DeviceType.CPU, enums.ShiftType.CIRC
    )
    tutils.check_or_record_results(
        shifted_projections,
        test_name,
        test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results
    )


def test_circ_shift_class_cpu(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_circ_shift_class_gpu"
    comparison_test_name = "test_circ_shift_class_cpu"
    complex_projections = tutils.prepare_data(filename)
    if repeat_array:
        tutils.repeat_array(complex_projections, n_reps)
    shifted_projections = shift_projections(
        complex_projections, enums.DeviceType.GPU, enums.ShiftType.CIRC
    )
    tutils.check_or_record_results(
        shifted_projections,
        test_name,
        comparison_test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results
    )


def test_fft_shift_class_gpu_chunked(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_fft_shift_class_gpu_chunked"
    comparison_test_name = "test_fft_shift_class_cpu"
    complex_projections = tutils.prepare_data(filename)
    if repeat_array:
        tutils.repeat_array(complex_projections, n_reps)
    t0 = time.time()
    shifted_projections = shift_projections(
        complex_projections, enums.DeviceType.GPU, enums.ShiftType.FFT, True
    )
    print(test_name, time.time() - t0)
    tutils.check_or_record_results(
        shifted_projections,
        test_name,
        comparison_test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results
    )


def test_circ_shift_class_gpu_chunked(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_circ_shift_class_gpu_chunked"
    comparison_test_name = "test_circ_shift_class_cpu"
    complex_projections = tutils.prepare_data(filename)
    if repeat_array:
        tutils.repeat_array(complex_projections, n_reps)
    shifted_projections = shift_projections(
        complex_projections, enums.DeviceType.GPU, enums.ShiftType.CIRC, True
    )
    tutils.check_or_record_results(
        shifted_projections,
        test_name,
        comparison_test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results
    )

def test_fft_shift_class_gpu_chunked_pinned(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_fft_shift_class_gpu_chunked_pinned"
    comparison_test_name = "test_fft_shift_class_cpu"
    complex_projections = tutils.prepare_data(filename)
    if repeat_array:
        tutils.repeat_array(complex_projections, n_reps)
        print(complex_projections.data.shape)
    complex_projections.pin_projections()
    assert is_pinned(complex_projections.data)
    t0 = time.time()
    shifted_projections = shift_projections(
        complex_projections, enums.DeviceType.GPU, enums.ShiftType.FFT, True
    )
    print(test_name, time.time() - t0)
    tutils.check_or_record_results(
        shifted_projections,
        test_name,
        comparison_test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results
    )

def test_fft_shift_class_gpu_chunked_multigpu(
    pytestconfig, overwrite_results=False, check_results=True
):
    if len(get_available_gpus()) < 5:
        pytest.skip("Not enough GPUs, skipping test")
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_fft_shift_class_gpu_chunked_multigpu"
    comparison_test_name = "test_fft_shift_class_cpu"
    complex_projections = tutils.prepare_data(filename)
    if repeat_array:
        tutils.repeat_array(complex_projections, n_reps)
        print(complex_projections.data.shape)
    complex_projections.pin_projections()
    assert is_pinned(complex_projections.data)
    t0 = time.time()
    shifted_projections = shift_projections(
        complex_projections,
        enums.DeviceType.GPU,
        enums.ShiftType.FFT,
        True,
        5,
        gpu_indices=(0, 1, 2, 3, 4),
        chunk_length=9,
    )
    print(test_name, time.time() - t0)
    tutils.check_or_record_results(
        shifted_projections,
        test_name,
        comparison_test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results
    )

def test_fft_shift_class_stay_on_gpu(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_fft_shift_class_stay_on_gpu"
    comparison_test_name = "test_fft_shift_class_cpu"
    complex_projections = tutils.prepare_data(filename)
    if repeat_array:
        tutils.repeat_array(complex_projections, n_reps)
    t0 = time.time()
    # shifted_projections = shift_projections(
    #     complex_projections,
    #     device_type=enums.DeviceType.GPU,
    #     shift_type=enums.ShiftType.FFT,
    #     chunking_enabled=False,
    # )

    n_images = len(complex_projections.angles)
    shift_array_size = (complex_projections.n_projections, 2)
    shift = (
        np.zeros(shift_array_size)
        + np.linspace(0, complex_projections.data.shape[2], n_images)[:, None]
    ).astype(r_type)

    shift_options = ShiftOptions()
    shift_options.enabled = True
    shift_options.device.device_type = enums.DeviceType.GPU,
    shift_options.type = enums.ShiftType.FFT
    shift_options.device.gpu.chunk_length = 100
    shift_options.device.gpu.chunking_enabled = False
    shifter = Shifter(shift_options)


    shifted_projections = shifter.run(
        cp.array(complex_projections.data), cp.array(shift)
    ).get()

    print(test_name, time.time() - t0)
    tutils.check_or_record_results(
        shifted_projections,
        test_name,
        comparison_test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    args = parser.parse_args()

    # args.skip_comparison=True

    test_fft_shift_class_cpu(None, args.overwrite_results, not args.skip_comparison)
    test_fft_shift_class_gpu(None, args.overwrite_results, not args.skip_comparison)
    test_circ_shift_class_cpu(None, args.overwrite_results, not args.skip_comparison)
    test_circ_shift_class_gpu(None, args.overwrite_results, not args.skip_comparison)
    test_fft_shift_class_gpu_chunked(None, args.overwrite_results, not args.skip_comparison)
    test_circ_shift_class_gpu_chunked(None, args.overwrite_results, not args.skip_comparison)
    test_fft_shift_class_gpu_chunked_pinned(None, args.overwrite_results, not args.skip_comparison)
    if len(get_available_gpus()) > 5:
        test_fft_shift_class_gpu_chunked_multigpu(None, args.overwrite_results, not args.skip_comparison)
    test_fft_shift_class_stay_on_gpu(None, args.overwrite_results, not args.skip_comparison)