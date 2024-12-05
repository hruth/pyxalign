import dataclasses
from typing import Sequence
from dataclasses import field
from llama.api.options.device import DeviceOptions


@dataclasses.dataclass
class FilterOptions:
    device: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class AstraOptions:
    back_project_gpu_indices: Sequence[int] = (0,)

    forward_project_gpu_indices: Sequence[int] = (0,)


@dataclasses.dataclass
class ReconstructOptions:
    astra: AstraOptions = field(default_factory=AstraOptions)

    filter: FilterOptions = field(default_factory=FilterOptions)
