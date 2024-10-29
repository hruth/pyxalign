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

class DownsampleType(StrEnum):
    FFT = auto()
    LINEAR = auto()