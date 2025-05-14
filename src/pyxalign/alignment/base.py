from abc import ABC, abstractmethod
from pyxalign.api.types import ArrayType
import numpy as np
import cupy as cp
from pyxalign.api.options.alignment import AlignmentOptions
import pyxalign.data_structures.projections as projections


class Aligner(ABC):
    def __init__(self, projections: "projections.Projections", options: AlignmentOptions):
        self.projections = projections  # this is a reference to the main dataset
        self.options = options
        self.shift = np.zeros((projections.n_projections, 2))
        # self.past_shifts = []

    @abstractmethod
    def run(self, *args, **kwargs) -> np.ndarray:
        pass

    # @abstractmethod
    # def calculate_alignment_shift(self, projections: ArrayType, *args, **kwargs) -> np.ndarray:
    #     pass
