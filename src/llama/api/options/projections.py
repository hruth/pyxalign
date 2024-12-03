from dataclasses import field
import llama.api.enums as enums
import dataclasses

from llama.api.options.options import ExperimentOptions, MaskOptions, PhaseUnwrapOptions
from llama.api.options.reconstruct import AstraReconstructOptions
from llama.api.options.transform import PreProcessingOptions


@dataclasses.dataclass
class ProjectionOptions:
    experiment: ExperimentOptions = field(default_factory=ExperimentOptions)

    astra: AstraReconstructOptions = field(default_factory=AstraReconstructOptions)

    mask: MaskOptions = field(default_factory=MaskOptions)

    phase_unwrap: PhaseUnwrapOptions = field(default_factory=PhaseUnwrapOptions)
    # Technically this should really only be here for complex projections

    # phase_ramp_removal: PhaseRampRemovalOptions = field(default_factory=PhaseRampRemovalOptions)
