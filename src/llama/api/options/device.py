import dataclasses
from dataclasses import field
from typing import Sequence

from llama.api import enums


@dataclasses.dataclass
class GPUOptions:
    chunking_enabled: bool = False

    chunk_length: int = 100

    n_gpus: int = 1

    gpu_indices: Sequence[int] = (0,)
    """The GPU indices to use for computation. If empty, use whatever GPUs are available in order."""


@dataclasses.dataclass
class DeviceOptions:
    device_type: enums.DeviceType = enums.DeviceType.CPU

    gpu: GPUOptions = field(default_factory=GPUOptions)
