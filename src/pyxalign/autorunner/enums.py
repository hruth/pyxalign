from csv import Error
from enum import StrEnum, auto
from typing import Union

class Checkpoints(StrEnum):
    INITIALIZATION = auto()
    CROSS_CORRELATION = auto()
    PHASE_UNWRAP_MASKS = auto()
    PHASE_UNWRAPPING = auto()
    RECONSTRUCTION_TUNING = auto()
    PMA_MASKS = auto()
    PROJECTION_MATCHING = auto()
    FINAL_RECONSTRUCTION = auto()

def get_checkpoint_order_value(checkpoint: Union[str, Checkpoints]) -> int:
    if checkpoint is None:
        return -1
    checkpoint = Checkpoints(checkpoint)
    # if checkpoint not in Checkpoints.__members__.keys():
    #     raise Error(f"""Checkpoint {checkpoint} not found. Did you add a new checkpoint?
    #                 If so, it needs to be added to the Checkpoints enum and to 
    #                 get_checkpoint_order_value.""")
    return {
        Checkpoints.INITIALIZATION: 1,
        Checkpoints.CROSS_CORRELATION: 2,
        Checkpoints.PHASE_UNWRAP_MASKS: 3,
        Checkpoints.PHASE_UNWRAPPING: 4,
        Checkpoints.RECONSTRUCTION_TUNING: 5,
        Checkpoints.PMA_MASKS: 6,
        Checkpoints.PROJECTION_MATCHING: 7,
        Checkpoints.FINAL_RECONSTRUCTION: 8,
    }[checkpoint]