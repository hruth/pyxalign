import argparse
import time
import h5py
import numpy as np
import pytest
from llama.api.options.device import DeviceOptions, GPUOptions

from llama.api.options.projections import ProjectionOptions
from llama.gpu_utils import is_pinned
from llama.projections import ComplexProjections
from llama.api.options.transform import DownsampleOptions, ShiftOptions
from llama.transformations.classes import Downsample, Shifter
from llama.api import enums

import llama.test_utils as tutils
from llama.api.types import ArrayType, r_type, c_type

filename = "cSAXS_projections_downsampling16.h5"
scale = 4


def downsample_projections(
    complex_projections: ComplexProjections,
    device_type: enums.DeviceType,
    downsample_type: enums.DownsampleType,
    chunking_enabled: bool = False,
    n_gpus: int = 1,
    gpu_indices: tuple = (0,),
    chunk_length: int = 100,
    apply_shift: bool = False,
    scale: int = scale,
):
    shift_array_size = (complex_projections.n_projections, 2)
    if apply_shift:
        n_images = len(complex_projections.angles)
        shift = (
            np.zeros(shift_array_size)
            + np.linspace(0, complex_projections.data.shape[2], n_images)[:, None]
        ).astype(r_type)
    else:
        shift = (np.zeros(shift_array_size)).astype(r_type)

    gpu_options = GPUOptions(
        chunking_enabled=chunking_enabled,
        chunk_length=chunk_length,
        n_gpus=n_gpus,
        gpu_indices=gpu_indices,
    )
    device_options = DeviceOptions(device_type=device_type, gpu_options=gpu_options)
    downsample_options = DownsampleOptions(
        type=downsample_type, scale=scale, enabled=True, device_options=device_options
    )
    downsampler = Downsample(downsample_options)
    processed_projections = downsampler.run(complex_projections.data, shift)

    return processed_projections


def check_size(complex_projections, processed_projections, scale):
    assert all(
        np.array(complex_projections.data.shape[1:]) / np.array(processed_projections.shape[1:])
        == scale
    )


def test_linear_downsample_class_cpu(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_linear_downsample_class_cpu"
    complex_projections = tutils.prepare_data(filename)

    processed_projections = downsample_projections(
        complex_projections=complex_projections,
        device_type=enums.DeviceType.CPU,
        downsample_type=enums.DownsampleType.LINEAR,
        chunking_enabled=False,
        apply_shift=True,
        scale=scale,
    )
    check_size(complex_projections, processed_projections, scale)
    assert complex_projections.data.dtype == processed_projections.dtype
    tutils.check_or_record_results(
        processed_projections,
        test_name,
        test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results,
    )

def test_linear_downsample_class_gpu(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_linear_downsample_class_gpu"
    comparison_test_name = "test_linear_downsample_class_cpu"
    complex_projections = tutils.prepare_data(filename)

    processed_projections = downsample_projections(
        complex_projections=complex_projections,
        device_type=enums.DeviceType.GPU,
        downsample_type=enums.DownsampleType.LINEAR,
        chunking_enabled=False,
        apply_shift=True,
        scale=scale,
    )
    check_size(complex_projections, processed_projections, scale)
    tutils.check_or_record_results(
        processed_projections,
        test_name,
        comparison_test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results,
    )

def test_linear_downsample_class_gpu_chunked(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_linear_downsample_class_gpu"
    comparison_test_name = "test_linear_downsample_class_cpu"
    complex_projections = tutils.prepare_data(filename)

    processed_projections = downsample_projections(
        complex_projections=complex_projections,
        device_type=enums.DeviceType.GPU,
        downsample_type=enums.DownsampleType.LINEAR,
        chunking_enabled=True,
        n_gpus=1,
        gpu_indices=(0, 1, 2, 3, 4),
        chunk_length=100,
        apply_shift=True,
        scale=scale,
    )
    check_size(complex_projections, processed_projections, scale)
    tutils.check_or_record_results(
        processed_projections,
        test_name,
        comparison_test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results,
    )

def test_linear_downsample_class_multigpu(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_linear_downsample_class_multigpu"
    comparison_test_name = "test_linear_downsample_class_cpu"
    complex_projections = tutils.prepare_data(filename)

    processed_projections = downsample_projections(
        complex_projections=complex_projections,
        device_type=enums.DeviceType.GPU,
        downsample_type=enums.DownsampleType.LINEAR,
        chunking_enabled=True,
        n_gpus=5,
        gpu_indices=(0, 1, 2, 3, 4),
        chunk_length=100,
        apply_shift=True,
        scale=scale,
    )
    check_size(complex_projections, processed_projections, scale)
    tutils.check_or_record_results(
        processed_projections,
        test_name,
        comparison_test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results,
    )

@pytest.mark.skip(reason="Skipping this test because I don't expect it to pass, but might want to check it later.")
def test_seperate_shift(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_seperate_shift"
    comparison_test_name = "test_linear_downsample_class_cpu"
    complex_projections = tutils.prepare_data(filename)

    processed_projections = downsample_projections(
        complex_projections=complex_projections,
        device_type=enums.DeviceType.GPU,
        downsample_type=enums.DownsampleType.LINEAR,
        chunking_enabled=True,
        n_gpus=5,
        gpu_indices=(0, 1, 2, 3, 4),
        chunk_length=100,
        apply_shift=True,
        scale=1,
    )
    processed_projections = ComplexProjections(
        processed_projections, complex_projections.angles, ProjectionOptions()
    )
    processed_projections = downsample_projections(
        complex_projections=processed_projections,
        device_type=enums.DeviceType.GPU,
        downsample_type=enums.DownsampleType.LINEAR,
        chunking_enabled=True,
        n_gpus=5,
        gpu_indices=(0, 1, 2, 3, 4),
        chunk_length=100,
        apply_shift=False,
        scale=scale,
    )
    check_size(complex_projections, processed_projections, scale)
    tutils.check_or_record_results(
        processed_projections,
        test_name,
        comparison_test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    args = parser.parse_args()

    # args.skip_comparison=True

    test_linear_downsample_class_cpu(None, args.overwrite_results, not args.skip_comparison)
    test_linear_downsample_class_gpu(None, args.overwrite_results, not args.skip_comparison)
    test_linear_downsample_class_multigpu(None, args.overwrite_results, not args.skip_comparison)
