from pyxalign.api.options.alignment import CrossCorrelationOptions, ProjectionMatchingOptions
import dataclasses
from dataclasses import field
from .base import BaseOptions


@dataclasses.dataclass
class AlignmentTaskOptions(BaseOptions):
    cross_correlation: CrossCorrelationOptions = field(default_factory=CrossCorrelationOptions)

    projection_matching: ProjectionMatchingOptions = field(default_factory=ProjectionMatchingOptions)