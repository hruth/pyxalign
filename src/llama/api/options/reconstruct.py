import dataclasses
from typing import Sequence


@dataclasses.dataclass
class AstraReconstructOptions:
    gpu_indices: Sequence[int] = ()