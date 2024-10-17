from abc import ABC, abstractmethod
import numpy as np
from llama.projections import Projections


class Aligner(ABC):
    def __init__(self, projections: Projections):
        self.projections = projections
        self.shift = np.zeros((projections.n_projections, 2))
        self.past_shifts = []

    @abstractmethod
    def get_alignment_shift(self):
        pass
        # raise NotImplementedError("Aligner classes must have get_alignment_shift")

    def _unstage_shift(self):
        if self.past_shifts != np.zeros((self.projections.n_projections, 2)):
            self.past_shifts += [self.staged_shift]
            self.staged_shift = np.zeros((self.projections.n_projections, 2))