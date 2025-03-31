from typing import Optional, Union
import h5py
import os
import numpy as np
from pathlib import Path

from llama.data_structures.projections import Projections
from llama.data_structures.task import LaminographyAlignmentTask
from llama.test_utils import ResultType, print_comparison_stats
from llama.api.options.tests import CITestOptions

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


class CITestHelper:
    def __init__(self, options: CITestOptions):
        self.parent_folder = os.path.join(
            os.environ["PYXALIGN_CI_TEST_DATA_DIR"], options.test_data_name
        )
        if not os.path.exists(self.parent_folder):
            raise FileNotFoundError(f"The folder {self.parent_folder} does not exist")
        self.inputs_folder, self.ci_results_folder, self.extra_results_folder = generate_ci_paths(
            self.parent_folder
        )
        self.options = options

    def save_or_compare_results(self, result, name: str):
        if self.options.update_tester_results:
            self.save_results(result, name)
        else:
            self.compare_results(result, name)

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
        print(f"{load_path} FAILED")
        print_comparison_stats(result, reference_result)
        raise AssertionError(f"Data for {name} does not match saved results!")
    else:
        print(f"{load_path} PASSED")


def save_arbitrary_result(result, folder: str, name: str, proj_idx: list[int]):
    file_path = os.path.join(folder, name + ".h5")
    with h5py.File(file_path, "w") as F:
        if isinstance(result, np.ndarray):
            save_array(result, F, array_save_string)
        elif isinstance(result, LaminographyAlignmentTask):
            save_task_ci(result, F, proj_idx)


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


def generate_ci_paths(ci_folder: str):
    inputs_folder = os.path.join(ci_folder, "inputs")
    ci_results_folder = os.path.join(ci_folder, "ci_results")
    extra_results_folder = os.path.join(ci_folder, "extra_results")

    for folder in [inputs_folder, ci_results_folder, extra_results_folder]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    return inputs_folder, ci_results_folder, extra_results_folder
