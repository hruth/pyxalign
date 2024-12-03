import dataclasses
from typing import Sequence


@dataclasses.dataclass
class AstraReconstructOptions:
    back_project_gpu_indices: Sequence[int] = (0,)

    forward_project_gpu_indices: Sequence[int] = (0,)
