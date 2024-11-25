from llama.api.options.alignment import CrossCorrelationOptions, ProjectionMatchingOptions
from llama.api.options.options import EstimateCenterOptions, PhaseRampRemovalOptions, MaskOptions
import dataclasses
from dataclasses import field
from llama.api.options.reconstruct import AstraReconstructOptions


@dataclasses.dataclass
class AlignmentTaskOptions:
    cross_correlation: CrossCorrelationOptions = field(default_factory=CrossCorrelationOptions)

    projection_matching: ProjectionMatchingOptions = field(default_factory=ProjectionMatchingOptions)

    astra_gpu: AstraReconstructOptions = field(default_factory=AstraReconstructOptions)

    mask: MaskOptions = field(default_factory=MaskOptions)

    estimate_center: EstimateCenterOptions = field(default_factory=EstimateCenterOptions)

    phase_ramp_removal: PhaseRampRemovalOptions = field(default_factory=PhaseRampRemovalOptions)
