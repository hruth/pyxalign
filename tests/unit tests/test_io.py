import os
import numpy as np

from llama.projections import PhaseProjections
from llama.task import LaminographyAlignmentTask
from llama.api import enums
from llama.io import save, load
from llama.api.options import ProjectionOptions, AlignmentTaskOptions


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
    task_options.projection_matching.regularization.local_TV = True
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
