from llama.api.options.alignment import CrossCorrelationOptions, ProjectionMatchingOptions
from llama.api.options.options import EstimateCenterOptions, ExperimentOptions, PhaseRampRemovalOptions, WeightsOptions
import dataclasses
from dataclasses import field
from llama.api.options.reconstruct import AstraReconstructOptions


@dataclasses.dataclass
class AlignmentTaskOptions:
    cross_correlation_options: CrossCorrelationOptions = field(
        default_factory=CrossCorrelationOptions
    )

    projection_matching_options: ProjectionMatchingOptions = field(
        default_factory=ProjectionMatchingOptions
    )

    astra_gpu_options: AstraReconstructOptions = field(default_factory=AstraReconstructOptions)

    weights_options: WeightsOptions = field(default_factory=WeightsOptions)

    estimate_center_options: EstimateCenterOptions = field(default_factory=EstimateCenterOptions)

    phase_ramp_removal_options: PhaseRampRemovalOptions = field(
        default_factory=PhaseRampRemovalOptions
    )