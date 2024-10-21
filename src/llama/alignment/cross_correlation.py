import numpy as np
from llama.alignment.base import Aligner
from llama.api.types import ArrayType
from llama.projections import Projections
from llama.api.options.alignment import CrossCorrelationOptions
from llama.transformations.functions import image_shift_circ


class CrossCorrelationAligner(Aligner):
    def run(self, options: CrossCorrelationOptions, projections: ArrayType, illum_sum: ArrayType):
        super.run(options, projections, illum_sum)

    def calculate_alignment_shift(self, projections: ArrayType, illum_sum: ArrayType) -> np.ndarray:
        pass
