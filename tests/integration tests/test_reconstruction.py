# import argparse
# import h5py
# import numpy as np

# from pyxalign.api.options.projections import ProjectionOptions
# from pyxalign.api.options.task import AlignmentTaskOptions
# from pyxalign.task import LaminographyAlignmentTask
# from pyxalign.projections import ComplexProjections
# from pyxalign.api import enums

# import pyxalign.test_utils as tutils

# file_name = "3a4bd4a_downsampled_4x.h5"

# def load_task(file_name):
#     task = load.load_task(file_path, exclude="complex_projections")

# def test_basic_reconstruction(pytestconfig, overwrite_results=False):
#     if pytestconfig is not None:
#         overwrite_results = pytestconfig.getoption("overwrite_results")

#     test_name = "test_basic_reconstruction"

#     load_task(file_name)

#     task.phase_projections.data = gutils.pin_memory(task.phase_projections.data)
#     task.phase_projections.masks = gutils.pin_memory(task.phase_projections.masks)

#     task.phase_projections.options.experiment.pixel_size = 2.74671658e-08 * scale
#     task.phase_projections.options.experiment.tilt_angle = 0
#     task.phase_projections.options.experiment.skew_angle = 0
#     rec = task.phase_projections.get_3D_reconstruction()

#     if overwrite_results:
#         tutils.save_results_data(shift, test_name, tutils.ResultType.SHIFT)
#     else:
#         tutils.compare_data(shift, test_name, tutils.ResultType.SHIFT)
#         tutils.print_passed_string(test_name)