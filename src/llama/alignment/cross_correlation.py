import numpy as np
from llama.projections import Projections
from llama.api.options.options import CrossCorrelationOptions
from llama.transformations import image_shift_circ


class CrossCorrelationAligner:
    def __init__(self, projections: Projections):
        self.projections = projections
        self.past_shifts = []

    def get_alignment_shift(self, projections: Projections, options: CrossCorrelationOptions):
        self.options = options
        # Cross correlation function executes here...
        self.staged_shift = np.zeros((self.projections.n_projections, 2))

    def run_cross_correlation_alignment(self):
        pass

    # def apply_shift(self, projections, shift):
    #     return image_shift_circ(projections, shift)
