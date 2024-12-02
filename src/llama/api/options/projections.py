from dataclasses import field
import llama.api.enums as enums
import dataclasses

from llama.api.options.options import ExperimentOptions, MaskOptions, PhaseUnwrapOptions
from llama.api.options.transform import PreProcessingOptions


@dataclasses.dataclass
class ProjectionDevice:
    pin_memory: bool = False

    device_type: enums.DeviceType = enums.DeviceType.CPU


@dataclasses.dataclass
class ProjectionOptions:
    experiment: ExperimentOptions = field(default_factory=ExperimentOptions)

    projection_device: ProjectionDevice = field(
        default_factory=ProjectionDevice
    )

    mask: MaskOptions = field(default_factory=MaskOptions)

    phase_unwrap: PhaseUnwrapOptions = field(default_factory=PhaseUnwrapOptions)
    # Technically this should really only be here for complex projections
