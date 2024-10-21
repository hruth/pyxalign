from abc import ABC, abstractmethod
from llama.api.types import ArrayType
import numpy as np
import cupy as cp
from llama.api.options.alignment import AlignmentOptions
from llama.api.options.transform import PreProcessingOptions
from llama.projections import Projections
from llama.src.llama.transformations.functions import image_pre_process
from llama.src.llama.transformations.classes import Downsample


class Aligner(ABC):
    def __init__(self, projections: Projections):
        self.projections = projections  # this is a reference to the main dataset
        self.shift = np.zeros((projections.n_projections, 2))
        self.past_shifts = []

    def run(self, options: AlignmentOptions):
        pre_processed_projections = pre_processed_projections = image_pre_process(
            self.projections.data, options.pre_processing_options
        )
        self.staged_shift = self.calculate_alignment_shift(pre_processed_projections)

    @abstractmethod
    def calculate_alignment_shift(self, projections: ArrayType) -> np.ndarray:
        pass

    def unstage_shift(self):
        if self.past_shifts != np.zeros((self.projections.n_projections, 2)):
            self.past_shifts += [self.staged_shift]
            self.staged_shift = np.zeros((self.projections.n_projections, 2))
