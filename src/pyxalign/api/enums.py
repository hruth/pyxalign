from enum import StrEnum, auto


class DeviceType(StrEnum):
    CPU = auto()
    GPU = auto()
    # __AUTOMATIC = auto()
    # "The device and device settings will automatically be chosen based on the input array type."


class ShiftType(StrEnum):
    FFT = auto()
    CIRC = auto()
    LINEAR = auto()


class RotationType(StrEnum):
    FFT = auto()


class ShearType(StrEnum):
    FFT = auto()


class DownsampleType(StrEnum):
    FFT = auto()
    LINEAR = auto()
    NEAREST = auto()


class UpsampleType(StrEnum):
    NEAREST = auto()


class SciPySubmodules(StrEnum):
    SIGNAL = auto()


class MemoryConfig(StrEnum):
    GPU_ONLY = auto()
    MIXED = auto()
    CPU_ONLY = auto()


class InputFileStructureType(StrEnum):
    LAMNI = auto()


class Direction(StrEnum):
    HORIZONTAL = auto()
    VERTICAL = auto()


class ProcessFunc(StrEnum):
    ANGLE = auto()
    ABS = auto()
    NONE = auto()


class SpecialValuePlaceholder(StrEnum):
    NONE = auto()
    EMPTY_LIST = auto()


class RoundType(StrEnum):
    CEIL = auto()
    FLOOR = auto()
    NEAREST = auto()

class ProjectionType(StrEnum):
    COMPLEX = auto()
    PHASE = auto()


class TestStartPoints(StrEnum):
    BEGINNING = auto()
    INITIAL_TASK = auto()
    PRE_PMA = auto()