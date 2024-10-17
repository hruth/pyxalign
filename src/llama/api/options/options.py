import dataclasses
from dataclasses import field
from typing import Sequence
import llama.api.enums as enums


@dataclasses.dataclass
class ExperimentOptions:
    laminography_angle: float = 61.1

    tilt_angle: float = -70.0

    skew_angle: float = 0

    sample_thickness: float = 7e-6


@dataclasses.dataclass
class GPUOptions:
    streams_per_gpu: int = 1

    chunks_per_gpu: int = 5

    n_gpus: int = 1

    gpu_indices: Sequence[int] = ()
    """The GPU indices to use for computation. If empty, use all available GPUs."""


@dataclasses.dataclass
class AstraReconstructOptions:
    gpu_indices: Sequence[int] = ()


@dataclasses.dataclass
class CrossCorrelationOptions:
    filter_position: int = 101

    filter_data: float = 0.005

    iterations: int = 100

    precision: float = 0.01

    downsampling: int = 4


@dataclasses.dataclass
class WeightsOptions:
    binary_close_coefficient: int = 30

    binary_erode_coefficient: int = 30

    unsharp: bool = True

    fill: int = 8


@dataclasses.dataclass
class ProjectionMatchingOptions:
    iterations: int = 300

    downsampling: int = 16

    high_pass_filter: float = 0.005

    step_relax: float = 0.1

    min_step_size: float = 0.01

    local_TV: bool = False

    local_TV_lambda: float = 3e-4

    gpu_options: GPUOptions = field(default_factory=GPUOptions)

    astra_gpu_options: AstraReconstructOptions = field(default_factory=GPUOptions)


@dataclasses.dataclass
class EstimateCenterOptions:
    downsampling: int = 8

    iterations: int = 2


@dataclasses.dataclass
class PhaseRampRemovalOptions:
    iterations: int = 5

    downsampling: int = 8


@dataclasses.dataclass
class AlignmentTaskOptions:
    cross_correlation_options: CrossCorrelationOptions = field(default_factory=CrossCorrelationOptions)

    projection_matching_options: CrossCorrelationOptions = field(default_factory=ProjectionMatchingOptions)

    astra_gpu_options: AstraReconstructOptions = field(default_factory=GPUOptions)


@dataclasses.dataclass
class ProjectionDeviceOptions:
    pin_memory: bool = False

    device_type: enums.DeviceType.CPU