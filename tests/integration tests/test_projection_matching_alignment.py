import argparse
import h5py
import numpy as np
from time import time

from llama.api.options.device import DeviceOptions, GPUOptions
from llama.api.options.projections import ProjectionOptions
from llama.api.options.task import AlignmentTaskOptions
from llama.task import LaminographyAlignmentTask
from llama.projections import ComplexProjections
from llama.api import enums

import llama.test_utils as tutils
import llama.gpu_utils as gutils

file_name = "task_3a4bd4a_downsampled_4x.h5"
n_iterations = 3
chunk_length = 20

def load_input_task():
    scale = 4
    task = tutils.load_task(file_name)

    task.phase_projections.data = gutils.pin_memory(task.phase_projections.data)
    task.phase_projections.masks = gutils.pin_memory(task.phase_projections.masks)
    task.phase_projections.options.experiment.pixel_size = 2.74671658e-08 * scale
    task.phase_projections.options.experiment.tilt_angle = 0
    task.phase_projections.options.experiment.skew_angle = 0

    return task

task = load_input_task()


def test_pma_mixed(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")

    test_name = "test_pma_mixed"
    comparison_test_name = "test_pma_mixed"

    task.options.projection_matching.iterations = n_iterations

    task.options.projection_matching.keep_on_gpu = False
    parent_gpu_settings = DeviceOptions(
        device_type=enums.DeviceType.GPU,
        gpu=GPUOptions(chunking_enabled=True, chunk_length=chunk_length),
    )
    task.options.projection_matching.device = parent_gpu_settings
    task.options.projection_matching.reconstruct.filter.device = parent_gpu_settings

    t0 = time()
    task.get_projection_matching_shift()
    print(time() - t0)
    shift = task.phase_projections.shift_manager.staged_shift

    assert task.pma_object.memory_config is enums.MemoryConfig.MIXED
    tutils.check_or_record_results(
        task.pma_object.reconstruction,
        test_name,
        comparison_test_name,
        overwrite_results,
        tutils.ResultType.RECONSTRUCTION,
        check_results,
    )


def test_pma_fully_on_gpu(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")

    test_name = "test_pma_fully_on_gpu"
    comparison_test_name = "test_pma_mixed"

    task.options.projection_matching.iterations = n_iterations

    task.options.projection_matching.keep_on_gpu = True
    parent_gpu_settings = DeviceOptions(
        device_type=enums.DeviceType.GPU,
        gpu=GPUOptions(chunking_enabled=True, chunk_length=chunk_length),
    )
    task.options.projection_matching.device = parent_gpu_settings
    task.options.projection_matching.reconstruct.filter.device = parent_gpu_settings

    t0 = time()
    task.get_projection_matching_shift()
    print(time() - t0)
    shift = task.phase_projections.shift_manager.staged_shift

    assert task.pma_object.memory_config is enums.MemoryConfig.GPU_ONLY
    tutils.check_or_record_results(
        task.pma_object.reconstruction,
        test_name,
        comparison_test_name,
        overwrite_results,
        tutils.ResultType.RECONSTRUCTION,
        check_results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-results", action="store_true")
    args = parser.parse_args()

    test_pma_mixed(None, overwrite_results=args.overwrite_results)
    test_pma_fully_on_gpu(None, overwrite_results=args.overwrite_results)
