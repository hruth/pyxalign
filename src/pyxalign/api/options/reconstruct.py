import dataclasses
from dataclasses import field
from typing import Optional
from pyxalign.api.enums import ReconstructionMethods, SARTInitialVolumes
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.options.options import RegularizationOptions
from .base import BaseOptions


@dataclasses.dataclass
class FilterOptions(BaseOptions):
    device: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class AstraOptions(BaseOptions):
    back_project_gpu_indices: tuple[int] = (0,)

    forward_project_gpu_indices: tuple[int] = (0,)

    algorithm_type: str = "BP3D_CUDA"


@dataclasses.dataclass
class GeometryOptions(BaseOptions):
    tilt_angle: float = 0.0

    skew_angle: float = 0.0


@dataclasses.dataclass
class SARTOptions(BaseOptions):
    iterations: int = 10

    use_circular_constraint: bool = True

    relaxation: float = 0  # from 0 to 1

    n_subtomograms: int = 1
    """
    Number of subtomograms to split the volume into.
    """

    initial_volume: SARTInitialVolumes = SARTInitialVolumes.ONES


def enabled_regularization_options_factory() -> RegularizationOptions:
    return RegularizationOptions(enabled=True)


@dataclasses.dataclass
class ReconstructOptions(BaseOptions):
    method: ReconstructionMethods = ReconstructionMethods.ASTRA

    astra: AstraOptions = field(default_factory=AstraOptions)

    geometry: GeometryOptions = field(default_factory=GeometryOptions)

    exclude_scans: Optional[list[int]] = None

    filter: FilterOptions = field(default_factory=FilterOptions)

    sart: SARTOptions = field(default_factory=SARTOptions)

    regularization: RegularizationOptions = field(
        default_factory=lambda: RegularizationOptions(enabled=True)
    )