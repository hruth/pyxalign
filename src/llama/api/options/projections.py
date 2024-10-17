from dataclasses import field
import llama.api.enums as enums
import dataclasses

from llama.api.options.transform import PreProcessingOptions


@dataclasses.dataclass
class ProjectionDeviceOptions:
    pin_memory: bool = False

    device_type: enums.DeviceType = enums.DeviceType.CPU


@dataclasses.dataclass
class ProjectionOptions:
    pre_processing_options: PreProcessingOptions = field(default_factory=PreProcessingOptions)

    projection_device_options: ProjectionDeviceOptions = field(
        default_factory=ProjectionDeviceOptions
    )