import os
import argparse
import time
import numpy as np
import cupy as cp
from llama.api.options.device import DeviceOptions
from llama.api.options.task import AlignmentTaskOptions
from llama.gpu_utils import is_pinned
from llama.projections import ComplexProjections, PhaseProjections
from llama.task import LaminographyAlignmentTask
from llama.transformations.classes import Shifter
from llama.api import enums
from llama.io import save, load

# import llama.api.options as opts
from llama.api.options.transform import ShiftOptions
from llama.api.options.projections import ProjectionOptions
from llama.api.options.options import ExperimentOptions
from llama.api.options.reconstruct import FilterOptions, ReconstructOptions

import llama.test_utils as tutils
from llama.api.types import r_type


def test_task_io(pytestconfig=None, overwrite_results=False, check_results=True):
    if pytestconfig is not None:
        overwrite_results = pytestconfig.getoption("overwrite_results")
    test_name = "test_task_io"

    # Create projection options
    projection_options = ProjectionOptions()
    # experiment
    projection_options.experiment.laminography_angle = 1
    projection_options.experiment.tilt_angle = 2
    projection_options.experiment.skew_angle = 3
    # reconstruct filter device
    projection_options.reconstruct.filter.device.device_type = enums.DeviceType.GPU
    projection_options.reconstruct.filter.device.gpu.n_gpus = 3
    projection_options.reconstruct.filter.device.gpu.gpu_indices = [3, 4, 1]
    projection_options.reconstruct.filter.device.gpu.chunking_enabled = True
    # Make dummy projections
    n_proj = 50
    projections = PhaseProjections(
        projections=np.random.rand(n_proj, 20, 20),
        angles=np.linspace(0, n_proj),
        options=projection_options,
        center_of_rotation=[123, 456],
        skip_pre_processing=True,
    )
    # Create task options
    task_options = AlignmentTaskOptions()
    task_options.projection_matching.iterations = 999
    task_options.projection_matching.regularization.local_TV = 789
    # Create task
    task = LaminographyAlignmentTask(options=task_options, phase_projections=projections)
    # save task
    file_name = "temp_task_file.h5"
    save.save_task(task, file_name)

    # load task
    loaded_task = load.load_task(file_name)

    # delete file
    if os.path.exists(file_name):
        os.remove(file_name)

    # Check if options are the same
    # check projection experiment options
    assert task.options == loaded_task.options
    assert projection_options.experiment == loaded_task.phase_projections.options.experiment
    assert np.allclose(task.phase_projections.data, loaded_task.phase_projections.data)


if __name__ == "__main__":
    test_task_io()
