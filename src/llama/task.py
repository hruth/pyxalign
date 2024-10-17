import numpy as np
from llama.projections import Projections
from llama.transformations import image_shift_circ, image_shift_fft
from llama.alignment.cross_correlation import CrossCorrelationAligner
from llama.alignment.projection_matching import ProjectionMatchingAligner
from llama.api.options.options import AlignmentTaskOptions


class LaminographyAlignmentTask:
    def __init__(self, projections: np.ndarray, options: AlignmentTaskOptions):
        self.projections = Projections(projections)
        self.options = options
        # need to think carefully about how the user can interact with the settings
        # one option would be to initialize everything in the beginning, and not allow
        # access to settings except through the individual methods. Or, have an update
        # method everytime objects are changed.
        self.cross_correlation_aligner = CrossCorrelationAligner()
        self.projection_matching_aligner = ProjectionMatchingAligner()

    def get_cross_correlation_shift(self):
        self.cross_correlation_aligner.get_alignment_shift()

    def apply_cross_correlation_shift(self):
        self.projections = image_shift_circ(
            self.projections, self.cross_correlation_aligner.staged_shift
        )
        self.cross_correlation_aligner._unstage_shift()

    def get_projection_matching_shift(self, downsampling: int, initial_shift: np.ndarray):
        self.projection_matching_aligner.get_alignment_shift(
            self.projections, self.options.projection_matching_options
        )

    def apply_projection_matching_shift(self):
        self.projections = image_shift_fft(
            self.projections, self.projection_matching_aligner.staged_shift
        )
        self.cross_correlation_aligner._unstage_shift()
