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