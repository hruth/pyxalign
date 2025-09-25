import dataclasses
from typing import Sequence
from dataclasses import field
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.options.options import ExperimentOptions


@dataclasses.dataclass
class FilterOptions:
    device: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class AstraOptions:
    back_project_gpu_indices: Sequence[int] = (0,)

    forward_project_gpu_indices: Sequence[int] = (0,)

    algorithm_type: str = "BP3D_CUDA"

@dataclasses.dataclass
class GeometryOptions:
    tilt_angle: float = 0.0

    skew_angle: float = 0.0

@dataclasses.dataclass
class ReconstructOptions:
    astra: AstraOptions = field(default_factory=AstraOptions)

    filter: FilterOptions = field(default_factory=FilterOptions)

    geometry: GeometryOptions = field(default_factory=GeometryOptions)
