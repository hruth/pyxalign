import h5py
from llama.api.options.projections import ProjectionOptions
from llama.api.options.task import AlignmentTaskOptions
from llama.task import LaminographyAlignmentTask
from llama.projections import ComplexProjections, PhaseProjections

filepath = '/home/beams/HRUTH/code/test_data/cSAXS_projections_downsampling16.h5'

# Open the HDF5 file for reading
with h5py.File(filepath, 'r') as h5file:
    # Read the datasets
    complex_projections = h5file['complex_projections'][:]
    angles = h5file['angles'][:]

projection_options = ProjectionOptions()
complex_projections = ComplexProjections(complex_projections, projection_options)

task_options = AlignmentTaskOptions()
task = LaminographyAlignmentTask(task_options)

