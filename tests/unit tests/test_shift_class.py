import argparse
import h5py
import numpy as np

from llama.api.options.projections import ProjectionOptions
from llama.projections import ComplexProjections
from llama.api.options.transform import ShiftOptions
from llama.transformations.classes import Shifter
from llama.api import enums

import llama.test_utils as tutils

filename = "cSAXS_projections_downsampling16.h5"


def prepare_data(
    device_type: enums.DeviceType, shift_type: enums.ShiftType, chunking_enabled: bool = False
) -> tuple[ComplexProjections, np.ndarray]:
    complex_projections, angles = tutils.load_input_projection_data(filename)

    projection_options = ProjectionOptions()
    complex_projections = ComplexProjections(complex_projections, angles, projection_options)

    n_images = len(angles)
    shift_array_size = (complex_projections.n_projections, 2)
    shift = (
        np.zeros(shift_array_size)
        + np.linspace(0, complex_projections.data.shape[2], n_images)[:, None]
    ).astype(np.float32)

    shift_options = ShiftOptions()
    shift_options.enabled = True
    shift_options.device_options.device_type = device_type
    shift_options.type = shift_type
    shift_options.device_options.gpu_options.chunk_size = 100
    shift_options.device_options.gpu_options.chunking_enabled = chunking_enabled
    shifter = Shifter(shift_options)
    shifted_projections = shifter.run(complex_projections.data, shift)

    return shifted_projections


def check_or_record_results(
    results: np.ndarray, test_name: str, comparison_test_name: str, overwrite_results: bool
):
    if overwrite_results:
        tutils.save_results_data(results, test_name, tutils.ResultType.PROJECTIONS_COMPLEX)
    else:
        tutils.compare_data(results, comparison_test_name, tutils.ResultType.PROJECTIONS_COMPLEX)
    tutils.print_passed_string(test_name)


def test_fft_shift_class_cpu(pytestconfig, overwrite_results=False, return_results=False):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_fft_shift_class_cpu"
    shifted_projections = prepare_data(enums.DeviceType.CPU, enums.ShiftType.FFT)
    check_or_record_results(shifted_projections, test_name, test_name, overwrite_results)


def test_fft_shift_class_gpu(pytestconfig, overwrite_results=False, return_results=False):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_fft_shift_class_gpu"
    comparison_test_name = "test_fft_shift_class_cpu"
    shifted_projections = prepare_data(enums.DeviceType.GPU, enums.ShiftType.FFT)
    check_or_record_results(shifted_projections, test_name, comparison_test_name, overwrite_results)


def test_circ_shift_class_gpu(pytestconfig, overwrite_results=False, return_results=False):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_circ_shift_class_cpu"
    shifted_projections = prepare_data(enums.DeviceType.CPU, enums.ShiftType.CIRC)
    check_or_record_results(shifted_projections, test_name, test_name, overwrite_results)


def test_circ_shift_class_cpu(pytestconfig, overwrite_results=False, return_results=False):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_circ_shift_class_gpu"
    comparison_test_name = "test_circ_shift_class_cpu"
    shifted_projections = prepare_data(enums.DeviceType.GPU, enums.ShiftType.CIRC)
    check_or_record_results(shifted_projections, test_name, comparison_test_name, overwrite_results)


def test_fft_shift_class_gpu_chunked(pytestconfig, overwrite_results=False, return_results=False):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_fft_shift_class_gpu_chunked"
    comparison_test_name = "test_fft_shift_class_cpu"
    shifted_projections = prepare_data(enums.DeviceType.GPU, enums.ShiftType.FFT, True)
    check_or_record_results(shifted_projections, test_name, comparison_test_name, overwrite_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-results", action="store_true")
    args = parser.parse_args()

    test_fft_shift_class_cpu(None, overwrite_results=args.overwrite_results)
    test_fft_shift_class_gpu(None, overwrite_results=args.overwrite_results)
    test_circ_shift_class_cpu(None, overwrite_results=args.overwrite_results)
    test_circ_shift_class_gpu(None, overwrite_results=args.overwrite_results)
    test_fft_shift_class_gpu_chunked(None, overwrite_results=args.overwrite_results)