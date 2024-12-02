import argparse
import h5py
import numpy as np

from llama.api.options.projections import ProjectionOptions
from llama.api.options.task import AlignmentTaskOptions
from llama.task import LaminographyAlignmentTask
from llama.projections import ComplexProjections
from llama.api import enums

import llama.test_utils as tutils

filename = "cSAXS_projections_downsampling16.h5"


def test_cross_correlation_shift(pytestconfig, overwrite_results=False):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")

    test_name = "test_cross_correlation_shift"

    complex_projections, angles = tutils.load_input_projection_data(filename)

    projection_options = ProjectionOptions()
    complex_projections = ComplexProjections(complex_projections, angles, projection_options)

    task_options = AlignmentTaskOptions()
    task = LaminographyAlignmentTask(task_options, complex_projections)

    task.get_cross_correlation_shift()
    task.apply_staged_shift()

    shift = task.shift_manager.past_shifts[0]
    if overwrite_results:
        tutils.save_results_data(shift, test_name, tutils.ResultType.SHIFT)
    else:
        tutils.compare_data(shift, test_name, tutils.ResultType.SHIFT)
        tutils.print_passed_string(test_name)


def test_cross_correlation_shift_gpu(pytestconfig, overwrite_results=False):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")

    test_name = "test_cross_correlation_shift_gpu"
    comparison_test_name = "test_cross_correlation_shift"

    complex_projections, angles = tutils.load_input_projection_data(filename)

    projection_options = ProjectionOptions()
    complex_projections = ComplexProjections(complex_projections, angles, projection_options)

    task_options = AlignmentTaskOptions()
    task_options.cross_correlation.device.device_type = enums.DeviceType.GPU
    task_options.cross_correlation.device.gpu.chunking_enabled = True
    task_options.cross_correlation.device.gpu.chunk_length = 10
    task = LaminographyAlignmentTask(task_options, complex_projections)

    task.get_cross_correlation_shift()
    task.apply_staged_shift()

    shift = task.shift_manager.past_shifts[0]

    tutils.check_or_record_results(
        shift, test_name, comparison_test_name, overwrite_results, tutils.ResultType.SHIFT
    )

    # if overwrite_results:
    #     tutils.save_results_data(shift, test_name, tutils.ResultType.SHIFT)
    # else:
    #     tutils.compare_data(shift, test_name, tutils.ResultType.SHIFT)
    #     tutils.print_passed_string(test_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-results", action="store_true")
    args = parser.parse_args()

    test_cross_correlation_shift(None, overwrite_results=args.overwrite_results)
    test_cross_correlation_shift_gpu(None, overwrite_results=args.overwrite_results)
