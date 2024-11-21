import argparse
import time
import h5py
import numpy as np

from llama.api.options.projections import ProjectionOptions
from llama.gpu_utils import is_pinned
from llama.projections import ComplexProjections
from llama.api.options.transform import ShiftOptions
from llama.transformations.classes import Shifter
from llama.api import enums

import llama.test_utils as tutils
from llama.api.types import ArrayType, r_type, c_type

filename = "cSAXS_projections_downsampling16.h5"
repeat_array = True
n_reps = 8


def shift_projections(
    complex_projections: ComplexProjections,
    device_type: enums.DeviceType,
    shift_type: enums.ShiftType,
    chunking_enabled: bool = False,
):

    n_images = len(complex_projections.angles)
    shift_array_size = (complex_projections.n_projections, 2)
    shift = (
        np.zeros(shift_array_size)
        + np.linspace(0, complex_projections.data.shape[2], n_images)[:, None]
    ).astype(r_type)

    shift_options = ShiftOptions()
    shift_options.enabled = True
    shift_options.device_options.device_type = device_type
    shift_options.type = shift_type
    shift_options.device_options.gpu_options.chunk_size = 100
    shift_options.device_options.gpu_options.chunking_enabled = chunking_enabled
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
    shifted_projections = shift_projections(
        complex_projections, enums.DeviceType.GPU, enums.ShiftType.FFT
    )
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
    complex_projections.pin_projections()
    print(is_pinned(complex_projections.data))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    args = parser.parse_args()

    # test_fft_shift_class_cpu(None, args.overwrite_results, not args.skip_comparison)
    # test_fft_shift_class_gpu(None, args.overwrite_results, not args.skip_comparison)
    # test_circ_shift_class_cpu(None, args.overwrite_results, not args.skip_comparison)
    # test_circ_shift_class_gpu(None, args.overwrite_results, not args.skip_comparison)
    test_fft_shift_class_gpu_chunked(None, args.overwrite_results, not args.skip_comparison)
    test_fft_shift_class_gpu_chunked(None, args.overwrite_results, not args.skip_comparison)
    # test_circ_shift_class_gpu_chunked(None, args.overwrite_results, not args.skip_comparison)

    test_fft_shift_class_gpu_chunked_pinned(None, args.overwrite_results, not args.skip_comparison)
    test_fft_shift_class_gpu_chunked_pinned(None, args.overwrite_results, not args.skip_comparison)

