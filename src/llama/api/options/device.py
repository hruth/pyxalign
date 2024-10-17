import dataclasses
from dataclasses import field
from typing import Sequence

from llama.api import enums


@dataclasses.dataclass
class GPUOptions:
    streams_per_gpu: int = 1

    chunks_per_gpu: int = 5

    n_gpus: int = 1

    gpu_indices: Sequence[int] = ()
    """The GPU indices to use for computation. If empty, use all available GPUs."""


@dataclasses.dataclass
class DeviceOptions:
    device_type: enums.DeviceType = enums.DeviceType.CPU

    gpu_options: GPUOptions = field(default_factory=GPUOptions)