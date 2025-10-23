import argparse
import traceback
from typing import Optional, Union
import h5py
import os
import numpy as np
from pathlib import Path
import subprocess

from pyxalign.data_structures.projections import Projections
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.data_structures.xrf_task import XRFTask
from pyxalign.io.save import save_array_as_tiff
from pyxalign.style.text import text_colors
from pyxalign.test_utils import print_comparison_stats
from pyxalign.data_structures.task import load_task
from pyxalign.api.options.tests import CITestOptions
from pyxalign.timing.io import get_timestamp_for_timing_files
from pyxalign.api.enums import TestStartPoints

projection_arrays_to_compare = [
    "data",
    "angles",
    "scan_numbers",
    "reconstructed_object_dimensions",
    # "dropped_scan_numbers",
    # "center_of_rotation",
    # "masks"
]
array_save_string = "array"
primary_ci_test_folder_string = "PYXALIGN_CI_TEST_DATA_DIR"
secondary_ci_test_folder_string = "PYXALIGN_CI_TEST_DATA_DIR_2"

pass_string = f"{text_colors.OKGREEN}PASSED{text_colors.ENDC}"
fail_string = f"{text_colors.FAIL}FAILED{text_colors.ENDC}"


class CITestHelper:
    def __init__(self, options: CITestOptions):
        self.options = options
        self.find_ci_test_folder()
        self.generate_ci_paths()
        self.store_run_metadata()
        self.test_result_dict = {}  # holds results of passes and fails when options.stop_on_error is false

    def find_ci_test_folder(self) -> str:
        for folder_string in [primary_ci_test_folder_string, secondary_ci_test_folder_string]:
            if folder_string in os.environ:
                ci_folder = os.path.join(
                    os.environ[folder_string],
                    self.options.test_data_name,
                )
                if os.path.exists(ci_folder) and os.path.exists(os.path.join(ci_folder, "inputs")):
                    self.parent_folder = os.path.join(ci_folder)
                    return

        raise FileNotFoundError(f"The input data folder for {self.options.test_data_name} was not found.")

    def generate_ci_paths(self):
        self.inputs_folder = os.path.join(self.parent_folder, "inputs")
        self.ci_results_folder = os.path.join(self.parent_folder, "ci_results")
        self.extra_results_folder = os.path.join(self.parent_folder, "extra_results")
        temp_results_folder = os.path.join(self.parent_folder, "temp_results")
        self.ci_temp_results_folder = os.path.join(temp_results_folder, "ci_results")
        self.extra_temp_results_folder = os.path.join(temp_results_folder, "extra_results")

        for folder in [
            self.inputs_folder,
            self.ci_results_folder,
            self.extra_results_folder,
            temp_results_folder,
            self.ci_temp_results_folder,
            self.extra_temp_results_folder,
        ]:
            if not os.path.exists(folder):
                os.mkdir(folder)

    def save_or_compare_results(
        self, result, name: str, atol: Optional[float] = None, rtol: Optional[float] = None
    ):
        if self.options.save_temp_files:
            self.save_temp_results(result, name)

        if self.options.update_tester_results:
            self.save_results(result, name)
        else:
            test_passed = False
            try:
                self.compare_results(result, name, atol, rtol)
                # the next line will only execute if an error is not
                # raised by compare_results
                test_passed = True
            except (AssertionError, ValueError, TypeError) as ex:
                if self.options.stop_on_error:
                    raise
                else:
                    print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
                    traceback.print_exc()
                    print("Continuing execution despite failed test")
            finally:
                self.test_result_dict[name] = test_passed

    def save_temp_results(self, result: str, name: str):
        save_arbitrary_result(
            result,
            self.ci_temp_results_folder,
            name,
            proj_idx=self.options.proj_idx,
        )

    def save_results(self, result: str, name: str):
        save_arbitrary_result(
            result,
            self.ci_results_folder,
            name,
            proj_idx=self.options.proj_idx,
        )

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
        compare_arbitrary_result(
            result, self.ci_results_folder, name, atol, rtol, self.options.proj_idx
        )

    def finish_test(self) -> Union[bool, None]:
        if not self.options.update_tester_results:
            print("SUMMARY OF TESTS:")
            for i, (test_name, is_passed) in enumerate(self.test_result_dict.items()):
                if is_passed:
                    pass_fail_string = pass_string
                else:
                    pass_fail_string = fail_string
                print(f"{i+1}. {test_name}: {pass_fail_string}")
            n_passed = sum([v for v in self.test_result_dict.values()])
            total_tests = len(self.test_result_dict)
            print(f"{text_colors.HEADER}{n_passed}/{total_tests}{text_colors.ENDC}")

            all_passed = n_passed == total_tests
            return all_passed

    def store_run_metadata(self):
        self.timestamp, date_string, time_string = get_timestamp_for_timing_files()

    def save_checkpoint_task(self, task: Union[LaminographyAlignmentTask, XRFTask], file_name: str):
        if self.options.update_tester_results:
            task.save_task(os.path.join(self.extra_results_folder, file_name))
        if self.options.save_temp_files:
            task.save_task(os.path.join(self.extra_temp_results_folder, file_name))

    def load_checkpoint_task(self, file_name: str) -> LaminographyAlignmentTask:
        return load_task(os.path.join(self.extra_results_folder, file_name))

    def save_tiff(
        self, array: np.ndarray, name: str, min: Optional[float] = None, max: Optional[float] = None
    ):
        if self.options.update_tester_results:
            save_array_as_tiff(array, os.path.join(self.extra_results_folder, name), min, max)
        if self.options.save_temp_files:
            save_array_as_tiff(array, os.path.join(self.extra_temp_results_folder, name), min, max)


class CITestArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # indicates start point of code
        self.parser.add_argument("--start-point", type=str, default=TestStartPoints.BEGINNING)
        # flag for specifying you want test results updated
        self.parser.add_argument("--update-results", action="store_true")
        self.parser.add_argument("--save-temp-results", action="store_true")
        self.parser.add_argument("--show-gui", action="store_true")


def compare_arbitrary_result(
    result,
    folder: str,
    name: str,
    atol=1e-3,
    rtol=1e-3,
    proj_idx: list[int] = [0],
):
    file_path = os.path.join(folder, name + ".h5")
    with h5py.File(file_path, "r") as F:
        if isinstance(result, np.ndarray):
            compare_arrays(result, F, array_save_string, atol, rtol)
        elif isinstance(result, LaminographyAlignmentTask):
            compare_tasks(result, F, atol, rtol, proj_idx)
        else:
            raise ValueError(
                f"{type(result).__qualname__} is not a supported type for CI comparisons"
            )


def compare_tasks(
    task: LaminographyAlignmentTask,
    h5_obj: Union[h5py.File, h5py.Group],
    atol: float,
    rtol: float,
    proj_idx: list[int],
):
    # define arrays to use in comparison
    if task.complex_projections is not None:
        compare_projections(
            task.complex_projections, h5_obj, "complex_projections", atol, rtol, proj_idx
        )
    if task.phase_projections is not None:
        compare_projections(
            task.phase_projections, h5_obj, "phase_projections", atol, rtol, proj_idx
        )


def compare_projections(
    projections: Projections,
    h5_obj: Union[h5py.File, h5py.Group],
    name: str,
    atol: float,
    rtol: float,
    proj_idx: list[int],
):
    for array_name in projection_arrays_to_compare:
        new_array = getattr(projections, array_name)
        if array_name == "data":
            new_array = new_array[proj_idx]
        compare_arrays(new_array, h5_obj, os.path.join(name, array_name), atol, rtol)


def compare_arrays(
    result: np.ndarray,
    h5_obj: Union[h5py.File, h5py.Group],
    name: str,
    atol: float,
    rtol: float,
):
    reference_result = h5_obj[name][()]
    load_path = get_rel_path_string(h5_obj[name])
    if not np.allclose(result, reference_result, atol=atol, rtol=rtol):
        print(f"{load_path} {fail_string}")
        print_comparison_stats(result, reference_result)
        raise AssertionError(f"Data for {name} does not match saved results!")
    else:
        print(f"{load_path} {pass_string}")


def save_arbitrary_result(result, folder: str, name: str, proj_idx: list[int]):
    file_path = os.path.join(folder, name + ".h5")
    with h5py.File(file_path, "w") as F:
        if isinstance(result, np.ndarray):
            save_array(result, F, array_save_string)
        elif isinstance(result, LaminographyAlignmentTask):
            save_task_ci(result, F, proj_idx)
        else:
            raise ValueError(
                f"{type(result).__qualname__} is not a supported type for CI comparisons"
            )


def save_task_ci(
    task: LaminographyAlignmentTask,
    h5_obj: Union[h5py.File, h5py.Group],
    proj_idx: list[int],
):
    if task.complex_projections is not None:
        save_projections_ci(task.complex_projections, h5_obj, "complex_projections", proj_idx)
    if task.phase_projections is not None:
        save_projections_ci(task.phase_projections, h5_obj, "phase_projections", proj_idx)


def save_projections_ci(
    projections: Projections,
    h5_obj: Union[h5py.File, h5py.Group],
    group_name: str,
    proj_idx: list[int],
):
    for array_name in projection_arrays_to_compare:
        array = getattr(projections, array_name)
        if array_name == "data":
            array = array[proj_idx]
        save_array(array, h5_obj, dataset_name=os.path.join(group_name, array_name))


def save_array(array: np.ndarray, h5_obj: Union[h5py.File, h5py.Group], dataset_name: str):
    h5_obj.create_dataset(dataset_name, data=array)
    save_path = get_rel_path_string(h5_obj[dataset_name])
    print(f'updated saved data for "{save_path}"')


def get_rel_path_string(h5_obj: h5py.Group):
    rel_path = Path(h5_obj.file.filename).name + h5_obj.name
    return rel_path
