from abc import ABC, abstractmethod
from llama.api.types import ArrayType
import numpy as np
import cupy as cp
from llama.api.options.alignment import AlignmentOptions
import llama.data_structures.projections as projections


class Aligner(ABC):
    def __init__(self, projections: "projections.Projections", options: AlignmentOptions):
        self.projections = projections  # this is a reference to the main dataset
        self.options = options
        self.shift = np.zeros((projections.n_projections, 2))
        # self.past_shifts = []

    @abstractmethod
    def run(self, *args, **kwargs) -> np.ndarray:
        pass

    # def run(self, *args, **kwargs):
    #     pre_processed_projections = PreProcess(self.options.pre_processing_options).run(
    #         self.projections.data
    #     )
        # self.staged_shift = self.calculate_alignment_shift(
        #     pre_processed_projections, self.projections.angles, *args, **kwargs
        # )

    # # How should arguments be specifed when implementing an abstract method?
    # # can an abstract method be wrapped? 
    # @abstractmethod
    # def run(self, options: AlignmentOptions, projections: ArrayType, *args, **kwargs):
    #     pass

    @abstractmethod
    def calculate_alignment_shift(self, projections: ArrayType, *args, **kwargs) -> np.ndarray:
        pass

    # def unstage_shift(self):
    #     if self.past_shifts != np.zeros((self.projections.n_projections, 2)):
    #         self.past_shifts += [self.staged_shift]
    #         self.staged_shift = np.zeros((self.projections.n_projections, 2))
