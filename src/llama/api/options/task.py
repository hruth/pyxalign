from llama.api.options.alignment import CrossCorrelationOptions, ProjectionMatchingOptions
import dataclasses
from dataclasses import field


@dataclasses.dataclass
class AlignmentTaskOptions:
    cross_correlation: CrossCorrelationOptions = field(default_factory=CrossCorrelationOptions)

    projection_matching: ProjectionMatchingOptions = field(default_factory=ProjectionMatchingOptions)