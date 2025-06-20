import argparse
import pytest
from pytest import Config
import pyxalign
import os
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import pyxalign.io.loaders.lamni.utils
import pyxalign.test_utils as tutils

try:
    from paths.lamni_loader_paths import parent_projection_folder, dat_file_path
except ModuleNotFoundError(""):
    pytest.skip("Skipping tests: file paths not found.", allow_module_level=True)


# Load data
if mp.cpu_count() >= 2:
    n_processes = mp.cpu_count() / 2
else:
    n_processes = 1


def test_lamni_loader(pytestconfig: Config, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_lamni_loader"

    options = pyxalign.io.loaders.lamni.utils.LYNXLoadOptions(
        selected_experiment_name="",
        selected_sequences=[3],
        selected_ptycho_strings=["512x512_b0_MLc_Niter500_recons"],
        scan_start=2720,
        scan_end=2750,
    )
    # Load data
    lamni_data = pyxalign.io.loaders.load_data_from_lamni_format(
        dat_file_path=dat_file_path,
        parent_projections_folder=parent_projection_folder,
        n_processes=int(mp.cpu_count() / 2),
        options=options,
    )

    assert (
        len(lamni_data.projections)
        == len(lamni_data.angles)
        == len(lamni_data.scan_numbers)
        == len(lamni_data.file_paths)
    )

    tutils.check_or_record_results(
        results=np.array([v[:400, :400] for v in lamni_data.projections.values()]),
        test_name=test_name,
        comparison_test_name=test_name,
        overwrite_results=overwrite_results,
        result_type=tutils.ResultType.PROJECTIONS_COMPLEX,
        check_results=check_results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    args = parser.parse_args()

    test_lamni_loader(None, args.overwrite_results, not args.skip_comparison)
