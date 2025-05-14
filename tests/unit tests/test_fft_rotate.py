import argparse
import time
import numpy as np
import cupy as cp
from pyxalign.gpu_utils import is_pinned
from pyxalign.data_structures.projections import ComplexProjections
from pyxalign.api.options.transform import ShiftOptions
from pyxalign.transformations.classes import Shifter
from pyxalign.api import enums

import pyxalign.test_utils as tutils
from pyxalign.api.types import r_type

filename = "cSAXS_projections_downsampling16.h5"
repeat_array = False
n_reps = 4


def test_fft_rotate_function(pytestconfig, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_fft_rotate"
    complex_projections = tutils.prepare_data(filename)
    # COMTINUE HERE

    tutils.check_or_record_results(
        rotated_projections,
        test_name,
        test_name,
        overwrite_results,
        tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-results", action="store_true")
    args = parser.parse_args()

    test_fft_rotate_function(args.overwrite_results)
