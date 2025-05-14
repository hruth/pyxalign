import h5py

from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.options.task import AlignmentTaskOptions
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.data_structures.projections import ComplexProjections, PhaseProjections

import matplotlib.pyplot as plt

filepath = '/home/beams/HRUTH/code/lamino_ci_test_data/cSAXS_projections_downsampling16.h5'

# Open the HDF5 file for reading
with h5py.File(filepath, 'r') as h5file:
    # Read the datasets
    complex_projections = h5file['complex_projections'][:]
    angles = h5file['angles'][:]

projection_options = ProjectionOptions()
complex_projections = ComplexProjections(complex_projections, angles, projection_options)

task_options = AlignmentTaskOptions()
task = LaminographyAlignmentTask(complex_projections, task_options)

task.get_cross_correlation_shift()

task.apply_cross_correlation_shift()

# Runs up until here
# To do: continue adding to and test cross-correlation code