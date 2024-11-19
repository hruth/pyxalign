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


def test_shift_class(pytestconfig, overwrite_results=False, return_results=False):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")

    test_name = "test_shift_class"

    complex_projections, angles = tutils.load_input_projection_data(filename)

    projection_options = ProjectionOptions()
    complex_projections = ComplexProjections(
        complex_projections, angles, projection_options
    )

    n_images = len(angles)
    shift_array_size = (complex_projections.n_projections, 2)
    shift = (
        np.zeros(shift_array_size)
        + np.linspace(0, complex_projections.data.shape[2], n_images)[:, None]
    ).astype(np.float32)

    shift_options = ShiftOptions()
    shift_options.enabled = True
    shifter = Shifter(shift_options)
    shifted_projections = shifter.run(complex_projections.data, shift)

    if overwrite_results:
        tutils.save_results_data(
            shifted_projections, test_name, tutils.ResultType.PROJECTIONS_COMPLEX
        )
    elif return_results:
        # For viewing results in jupyter notebooks
        return shifted_projections, complex_projections
    else:
        tutils.compare_data(
            shifted_projections, test_name, tutils.ResultType.PROJECTIONS_COMPLEX
        )


def test_shift_class_gpu(pytestconfig, overwrite_results=False, return_results=False):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_shift_class_gpu"
    comparison_test_name = "test_shift_class"

    complex_projections, angles = tutils.load_input_projection_data(filename)

    projection_options = ProjectionOptions()
    complex_projections = ComplexProjections(
        complex_projections, angles, projection_options
    )

    n_images = len(angles)
    shift_array_size = (complex_projections.n_projections, 2)
    shift = (
        np.zeros(shift_array_size)
        + np.linspace(0, complex_projections.data.shape[2], n_images)[:, None]
    ).astype(np.float32)

    shift_options = ShiftOptions()
    shift_options.enabled = True
    shift_options.device_options.device_type = enums.DeviceType.GPU
    shifter = Shifter(shift_options)
    shifted_projections = shifter.run(complex_projections.data, shift)

    if overwrite_results:
        tutils.save_results_data(
            shifted_projections, test_name, tutils.ResultType.PROJECTIONS_COMPLEX
        )
    if return_results:
        # For viewing results in jupyter notebooks
        return shifted_projections, complex_projections
    else:
        tutils.compare_data(
            shifted_projections,
            comparison_test_name,
            tutils.ResultType.PROJECTIONS_COMPLEX,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-results", action="store_true")
    args = parser.parse_args()

    # test_shift_class(None, overwrite_results=args.overwrite_results)
    test_shift_class_gpu(None, overwrite_results=args.overwrite_results)
