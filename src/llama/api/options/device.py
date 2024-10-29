import dataclasses
from dataclasses import field
from typing import Sequence

from llama.api import enums


@dataclasses.dataclass
class GPUOptions:
    streams_per_gpu: int = 1  # Maybe it is better to remove this while only using a single stream.

    chunks_per_gpu: int = 1

    n_gpus: int = 1

    gpu_indices: Sequence[int] = (0,)
    """The GPU indices to use for computation. If empty, use all available GPUs."""


@dataclasses.dataclass
class DeviceOptions:
    device_type: enums.DeviceType = enums.DeviceType.CPU

    gpu_options: GPUOptions = field(default_factory=GPUOptions)
