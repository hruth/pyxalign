from csv import Error
from enum import StrEnum, auto
from typing import Union


class AutorunnerStep(StrEnum):
    AUTORUNNER_CONFIGURATION_WINDOW = "Autorunner Configuration Window"
    DATA_LOADER_WINDOW = "Data Loading and Initialization Window"
    COMPLEX_PROJECTIONS_WINDOW = "Complex Projections Window"
    PHASE_UNWRAPPING_WINDOW = "Phase Unwrapping Window"
    UNWRAPPED_PROJECTIONS_WINDOW = "Unwrapped Projections Window"


class Checkpoints(StrEnum):
    # if you add to these, or change these, make sure to also update the
    # CheckpointsConfig in the config.py file
    AFTER_LOADING = auto()
    AFTER_COMPLEX_PROJECTIONS_WINDOW = auto()
    AFTER_PHASE_UNWRAPPING_WINDOW = auto()
    FINAL = auto()


class LoadableCheckpoints(StrEnum):
    # has all members of checkpoints, except for FINAL
    AFTER_LOADING = auto()
    AFTER_COMPLEX_PROJECTIONS_WINDOW = auto()
    AFTER_PHASE_UNWRAPPING_WINDOW = auto()


def get_checkpoint_order_value(checkpoint: Union[str, Checkpoints]) -> int:
    if checkpoint is None:
        return -1
    checkpoint = Checkpoints(checkpoint)
    # if checkpoint not in Checkpoints.__members__.keys():
    #     raise Error(f"""Checkpoint {checkpoint} not found. Did you add a new checkpoint?
    #                 If so, it needs to be added to the Checkpoints enum and to 
    #                 get_checkpoint_order_value.""")
    return {
        Checkpoints.AFTER_LOADING: 1,
        Checkpoints.AFTER_COMPLEX_PROJECTIONS_WINDOW: 2,
        Checkpoints.AFTER_PHASE_UNWRAPPING_WINDOW: 3,
        Checkpoints.FINAL: 4,
    }[checkpoint]