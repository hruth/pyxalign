from dataclasses import field
import llama.api.enums as enums
import dataclasses
from llama.api.options.device import DeviceOptions

from llama.api.options.options import ExperimentOptions, MaskOptions, PhaseUnwrapOptions
from llama.api.options.reconstruct import ReconstructOptions
from llama.api.options.transform import CropOptions, DownsampleOptions


@dataclasses.dataclass
class ProjectionOptions:
    experiment: ExperimentOptions = field(default_factory=ExperimentOptions)

    reconstruct: ReconstructOptions = field(default_factory=ReconstructOptions)

    mask: MaskOptions = field(default_factory=MaskOptions)

    # Technically this should really only be here for complex projections
    phase_unwrap: PhaseUnwrapOptions = field(default_factory=PhaseUnwrapOptions)

    crop: CropOptions = field(default_factory=CropOptions)

    downsample: DownsampleOptions = field(default_factory=DownsampleOptions)

    mask_downsample_type: enums.DownsampleType = enums.DownsampleType.LINEAR

    # phase_ramp_removal: PhaseRampRemovalOptions = field(default_factory=PhaseRampRemovalOptions)
