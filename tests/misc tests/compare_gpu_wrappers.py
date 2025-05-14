import argparse
import h5py
import numpy as np
import copy
import astra

from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.options.task import AlignmentTaskOptions
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.data_structures.projections import ComplexProjections
from pyxalign.api import enums
from pyxalign import reconstruct
import demo_utils as dutils
import pyxalign.gpu_utils as gutils
import pyxalign.api.options as opts

import laminoAlign as lam

import pyxalign.test_utils as tutils

def filter_sinogram_new():
    # Load demo task (750 x 960 x 2368 projections)
    task = dutils.load_demo_task(
        file_name="unaligned_task.h5",
        sub_folder="laminoAlign_preprocessed_data",
        use_local=True,
    )

    volume = task.phase_projections.volume
    # Get vectors needed for initialization
    volume.options = copy.deepcopy(volume.projections.options.reconstruct)
    volume.experiment_options = copy.deepcopy(volume.projections.options.experiment)
    astra.set_gpu_index(volume.options.astra.back_project_gpu_indices)
    scan_geometry_config, vectors = reconstruct.get_astra_reconstructor_geometry(
        size=volume.projections.data,
        angles=volume.projections.angles,
        n_pix=volume.projections.reconstructed_object_dimensions,
        center_of_rotation=volume.projections.center_of_rotation,
        lamino_angle=volume.experiment_options.laminography_angle,
        tilt_angle=volume.experiment_options.tilt_angle,
        skew_angle=volume.experiment_options.skew_angle,
    )
    # Pin memory for filtering
    pinned_filtered_sinogram = gutils.pin_memory(np.zeros_like(task.phase_projections.data))
    task.phase_projections.data = gutils.pin_memory(task.phase_projections.data)
    # Filter sinogram
    gpu_list = gutils.get_available_gpus()
    device_options = opts.DeviceOptions(
        device_type=enums.DeviceType.GPU,
        gpu=opts.GPUOptions(
            chunk_length=20,
            chunking_enabled=True,
            n_gpus=gpu_list,
            gpu_indices=gpu_list,
        ),
    )
    sinogram = reconstruct.filter_sinogram(
        sinogram=task.phase_projections.data,
        vectors=vectors,
        device_options=device_options,
        pinned_results=pinned_filtered_sinogram,
    )

    print(gutils.is_pinned(sinogram))
    print(gutils.is_pinned(pinned_filtered_sinogram))
    print(gutils.is_pinned(task.phase_projections.data))
    print(pinned_filtered_sinogram is sinogram)


def filter_sinogram_old():
    lam.FBP.filterSinogram

if __name__ == "__main__":
    filter_sinogram_new()