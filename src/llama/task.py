import numpy as np
from llama.projections import ComplexProjections, PhaseProjections
from llama.src.llama.transformations.functions import image_shift_fft
from llama.src.llama.transformations.functions import image_shift_circ
from llama.alignment.cross_correlation import CrossCorrelationAligner
from llama.alignment.projection_matching import ProjectionMatchingAligner
from llama.api.options.task import AlignmentTaskOptions


class LaminographyAlignmentTask:
    def __init__(
        self, complex_projections: ComplexProjections, options: AlignmentTaskOptions
    ):  # maybe change it to take in a projections objects
        self.complex_projections = complex_projections
        self.options = options
        self.cross_correlation_aligner = CrossCorrelationAligner()

    def get_cross_correlation_shift(self):
        self.cross_correlation_aligner.run(self.options.cross_correlation_options, self.illum_sum)

    def apply_cross_correlation_shift(self):
        self.projections = image_shift_circ(
            self.projections, self.cross_correlation_aligner.staged_shift
        )
        self.cross_correlation_aligner.unstage_shift()
