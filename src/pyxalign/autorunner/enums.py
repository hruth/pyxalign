from enum import StrEnum, auto
from typing import Union

class Checkpoints(StrEnum):
    INITIALIZATION = auto()
    CROSS_CORRELATION_ALIGNMENT = auto()
    PHASE_UNWRAPPING = auto()
    PHASE_PROJECTIONS_MASKS = auto()
    ESTIMATE_CENTER = auto()

def get_checkpoint_order_value(checkpoint: Union[str, Checkpoints]) -> int:
    checkpoint = Checkpoints(checkpoint)
    return {
        Checkpoints.INITIALIZATION: 1,
        Checkpoints.CROSS_CORRELATION_ALIGNMENT: 2,
        Checkpoints.PHASE_UNWRAPPING: 3,
        Checkpoints.PHASE_PROJECTIONS_MASKS: 4,
        Checkpoints.ESTIMATE_CENTER: 5,
    }[checkpoint]