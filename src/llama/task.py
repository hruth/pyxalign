import numpy as np
from llama.projections import ComplexProjections, PhaseProjections, Projections
from llama.transformations.functions import image_shift_fft
from llama.transformations.functions import image_shift_circ
from llama.alignment.cross_correlation import CrossCorrelationAligner
from llama.alignment.projection_matching import ProjectionMatchingAligner
from llama.api.options.task import AlignmentTaskOptions


class LaminographyAlignmentTask:
    def __init__(
        self, projections: Projections, options: AlignmentTaskOptions
    ):  # update later to allow complex or phase projections
        self.options = options
        self.projections = projections
        self.cross_correlation_aligner = CrossCorrelationAligner(
            self.projections, self.options.cross_correlation_options
        )
        self.past_shifts = [] # for storing the shifts that have been applied

    def get_cross_correlation_shift(self):
        self.illum_sum = np.ones_like(self.projections.data[0])  # Temporary
        self.cross_correlation_aligner.run(self.illum_sum)

    def apply_cross_correlation_shift(self):
        self.projections.set_data(
            image_shift_circ(self.projections, self.cross_correlation_aligner.staged_shift)
        )
        self.cross_correlation_aligner.unstage_shift()

    def _unstage_shift(self):
        if self.staged_shift != np.zeros((self.projections.n_projections, 2)):
            self.past_shifts += [self.staged_shift]
            self.staged_shift = np.zeros((self.projections.n_projections, 2))