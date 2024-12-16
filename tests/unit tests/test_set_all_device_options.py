
from llama.api.options_utils import set_all_device_options
import llama.api.options as opts
import llama.api.enums as enums

new_options = opts.DeviceOptions(
    device_type=enums.DeviceType.GPU,
    gpu=opts.GPUOptions(chunk_length=123, chunking_enabled=True),
)
task_options = opts.AlignmentTaskOptions()
set_all_device_options(task_options, new_options)
print(task_options.projection_matching.reconstruct.filter.device)
print(task_options.projection_matching.device)