import h5py
import os
import numpy as np
from enum import StrEnum, auto
from llama.api.options.projections import ProjectionOptions

from llama.projections import ComplexProjections


class ResultType(StrEnum):
    SHIFT = auto()
    PROJECTIONS_PHASE = auto()
    PROJECTIONS_AMPLITUDE = auto()
    PROJECTIONS_COMPLEX = auto()


def get_ci_data_dir():
    try:
        dir = os.environ["LAMINO_ALIGN_DATA_DIR"]
    except KeyError:
        raise KeyError(
            "LAMINO_ALIGN_DATA_DIR not set. Please set it to the path to the data folder."
        )
    return dir


def get_ci_input_data_dir():
    return os.path.join(get_ci_data_dir(), "input_data")


def get_ci_results_dir():
    return os.path.join(get_ci_data_dir(), "test_results")


def generate_results_folder(test_name):
    return os.path.join(get_ci_results_dir(), test_name)


def generate_results_path(test_name: str, variable_type: ResultType):
    return os.path.join(generate_results_folder(test_name), variable_type + ".npy")


def save_results_data(data: np.ndarray, test_name: str, variable_type: ResultType):
    os.makedirs(generate_results_folder(test_name), exist_ok=True)
    filepath = generate_results_path(test_name, variable_type)
    np.save(filepath, data)
    print("Data saved to " + filepath)


def load_input_projection_data(filename: str) -> tuple[np.ndarray, np.ndarray]:
    filepath = os.path.join(get_ci_input_data_dir(), filename)
    with h5py.File(filepath, "r") as h5file:
        complex_projections = h5file["complex_projections"][:]
        angles = h5file["angles"][:]
    return complex_projections, angles


def compare_data(data: np.ndarray, comparison_test_name: str, variable_type: ResultType, atol=1e-3, rtol=1e-3):
    filepath = generate_results_path(comparison_test_name, variable_type)
    old_data = np.load(filepath)
    if not np.allclose(data, old_data, atol=atol, rtol=rtol):
        raise AssertionError
    
def print_passed_string(test_name: str):
    print("{} PASSED".format(test_name))

def check_or_record_results(
    results: np.ndarray, test_name: str, comparison_test_name: str, overwrite_results: bool, result_type: ResultType
):
    if overwrite_results:
        save_results_data(results, test_name, result_type)
    else:
        compare_data(results, comparison_test_name, result_type)
    print_passed_string(test_name)

def prepare_data(filename) ->ComplexProjections:
    complex_projections, angles = load_input_projection_data(filename)
    projection_options = ProjectionOptions()
    complex_projections = ComplexProjections(complex_projections, angles, projection_options)
    return complex_projections