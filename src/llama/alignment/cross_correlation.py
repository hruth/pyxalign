import numpy as np
from llama.alignment.base import Aligner
from llama.api.types import ArrayType
from llama.projections import Projections
from src.llama.api.options.alignment import CrossCorrelationOptions
from llama.transformations import image_shift_circ


class CrossCorrelationAligner(Aligner):
    def calculate_alignment_shift(self, projections: ArrayType) -> np.ndarray:
        self.staged_shift = np.zeros((self.projections.n_projections, 2))
