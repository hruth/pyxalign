from enum import StrEnum, auto

class DeviceType(StrEnum):
    CPU = auto()
    GPU = auto()

class ShiftType(StrEnum):
    FFT = auto()
    CIRC = auto()
    LINEAR = auto()

class DownsampleType(StrEnum):
    FFT = auto()
    LINEAR = auto()