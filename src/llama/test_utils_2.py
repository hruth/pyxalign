from typing import Any, Optional
import h5py
import os
import numpy as np
from enum import StrEnum, auto
from llama.api.options.projections import ProjectionOptions

from llama.data_structures.projections import ComplexProjections
from llama.data_structures.task import LaminographyAlignmentTask
from llama.io import load
from llama.test_utils import ResultType, print_comparison_stats
from llama.api.options.tests import CITestOptions


class CITestHelper:
    def __init__(self, options: CITestOptions):
        self.parent_folder = os.path.join(
            os.environ["PYXALIGN_CI_TEST_DATA_DIR"], options.test_data_name
        )
        if not os.path.exists(self.parent_folder):
            raise FileNotFoundError(f"The folder {self.parent_folder} does not exist")
        self.inputs_folder, self.results_folder = generate_ci_paths(self.parent_folder)
        self.options = options

    def save_or_compare_results(self, result, name: str):
        if self.options.update_tester_results:
            self.save_results(result, name)
        else:
            self.compare_results(result, name)

    def save_results(self, result: str, name: str):
        save_arbitrary_result(result, self.results_folder, name)

    def compare_results(
        self,
        result: str,
        name: str,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        if atol is None:
            atol = self.options.atol
        if rtol is None:
            rtol = self.options.rtol
        compare_arbitrary_result(result, self.results_folder, name, atol, rtol)


def compare_arbitrary_result(result, folder: str, name: str, atol=1e-3, rtol=1e-3):
    if isinstance(result, np.ndarray):
        ext = ".h5"
        file_path = os.path.join(folder, name) + ext
        with h5py.File(file_path, "r") as F:
            reference_result = F["data"][()]
        print(f"comparison of {name}")
        print_comparison_stats(result, reference_result)
        if not np.allclose(result, reference_result, atol=atol, rtol=rtol):
            raise AssertionError(f"Data for {name} does not match")


def save_arbitrary_result(result, folder: str, name: str):
    if isinstance(result, np.ndarray):
        ext = ".h5"
        file_path = os.path.join(folder, name) + ext
        with h5py.File(file_path, "w") as F:
            F.create_dataset("data", data=result)
    print(f"updated saved data for \"{name}\"")


def generate_ci_paths(ci_folder: str):
    inputs_folder = os.path.join(ci_folder, "inputs")
    results_folder = os.path.join(ci_folder, "results")

    for folder in [inputs_folder, results_folder]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    return inputs_folder, results_folder
