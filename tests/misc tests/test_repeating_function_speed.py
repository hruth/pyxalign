import argparse
import h5py
import numpy as np
from time import time

from llama.api.options.device import DeviceOptions, GPUOptions
from llama.api.options.projections import ProjectionOptions
from llama.api.options.reconstruct import ReconstructOptions
from llama.api.options.task import AlignmentTaskOptions
from llama.data_structures.task import LaminographyAlignmentTask
from llama.data_structures.projections import ComplexProjections
from llama.api import enums
from llama.api.enums import DownsampleType

import llama.test_utils as tutils
import llama.gpu_utils as gutils

file_name = "task_3a4bd4a_downsampled_4x.h5"
n_iterations = 3
chunk_length = 20


def load_input_task() -> LaminographyAlignmentTask:
    scale = 4
    task = tutils.load_task(file_name)

    task.phase_projections.data = gutils.pin_memory(task.phase_projections.data)
    task.phase_projections.masks = gutils.pin_memory(task.phase_projections.masks)
    task.phase_projections.pixel_size = 2.74671658e-08 * scale
    task.phase_projections.options.experiment.tilt_angle = 0
    task.phase_projections.options.experiment.skew_angle = 0

    return task


def use_all_gpus_for_astra(reconstruct_options: ReconstructOptions):
    gpu_indices = gutils.get_available_gpus()
    reconstruct_options.astra.forward_project_gpu_indices = gpu_indices
    reconstruct_options.astra.back_project_gpu_indices = (0,)


def run_pma():
    task = load_input_task()

    # PMA settings
    n_iterations = 300
    task.options.projection_matching.iterations = n_iterations
    task.options.projection_matching.downsample.enabled = False
    task.options.projection_matching.downsample.scale = 4
    task.options.projection_matching.downsample.type = DownsampleType.LINEAR

    # Device settings
    task.options.projection_matching.keep_on_gpu = False
    parent_gpu_settings = DeviceOptions(
        device_type=enums.DeviceType.GPU,
        gpu=GPUOptions(chunking_enabled=True, chunk_length=chunk_length),
    )
    task.options.projection_matching.device = parent_gpu_settings
    task.options.projection_matching.reconstruct.filter.device = parent_gpu_settings

    use_all_gpus_for_astra(task.options.projection_matching.reconstruct)

    task.get_projection_matching_shift()


if __name__ == "__main__":
    run_pma()
