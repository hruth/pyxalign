from pyxalign import options as opts
from pyxalign.test_utils_2 import CITestHelper

# Setup the test
# ci_options = opts.CITestOptions(test_data_name="cSAXS_e18044_LamNI_201907")
ci_options = opts.CITestOptions(test_data_name="TP2")
ci_test_helper = CITestHelper(options=ci_options)
# load data with phase unwrapped projections
task = ci_test_helper.load_checkpoint_task(file_name="pre_pma_task.h5")
# launch viewer
task.phase_projections.launch_viewer(wait_until_closed=True)

