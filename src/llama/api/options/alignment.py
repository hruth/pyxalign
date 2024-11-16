from abc import ABC, abstractmethod
import dataclasses
from dataclasses import field
from llama.api.options.device import DeviceOptions, GPUOptions
from llama.api.options.reconstruct import AstraReconstructOptions
from llama.api.options.transform import PreProcessingOptions


@dataclasses.dataclass
class AlignmentOptions(ABC):
    gpu_options: GPUOptions = field(default_factory=GPUOptions)

    pre_processing_options: PreProcessingOptions = field(default_factory=PreProcessingOptions)

    device_options: DeviceOptions = field(default_factory=DeviceOptions)
    # @property
    # @abstractmethod
    # def gpu_options(self) -> GPUOptions:
    #     pass

    # @property
    # @abstractmethod
    # def pre_processing_options(self) -> PreProcessingOptions:
    #     pass


@dataclasses.dataclass
class CrossCorrelationOptions(AlignmentOptions):
    iterations: int = 100

    binning: int = 4

    filter_position: int = 101

    filter_data: float = 0.005

    precision: float = 0.01

    # Inherited and overwritten follows:
    device_options: DeviceOptions = field(default_factory=DeviceOptions)

    gpu_options: GPUOptions = field(default_factory=GPUOptions)

    pre_processing_options: PreProcessingOptions = field(default_factory=PreProcessingOptions)


@dataclasses.dataclass
class ProjectionMatchingOptions(AlignmentOptions):
    iterations: int = 300

    high_pass_filter: float = 0.005

    step_relax: float = 0.1

    min_step_size: float = 0.01

    local_TV: bool = False

    local_TV_lambda: float = 3e-4

    astra_gpu_options: AstraReconstructOptions = field(default_factory=GPUOptions)

    # Inherited and overwritten follows:
    device_options: DeviceOptions = field(default_factory=DeviceOptions)

    gpu_options: GPUOptions = field(default_factory=GPUOptions)

    pre_processing_options: PreProcessingOptions = field(default_factory=PreProcessingOptions)
