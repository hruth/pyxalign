import dataclasses


@dataclasses.dataclass
class ExperimentOptions:
    laminography_angle: float = 61.1

    tilt_angle: float = -70.0

    skew_angle: float = 0

    sample_thickness: float = 7e-6


@dataclasses.dataclass
class WeightsOptions:
    binary_close_coefficient: int = 30

    binary_erode_coefficient: int = 30

    unsharp: bool = True

    fill: int = 8


@dataclasses.dataclass
class EstimateCenterOptions:
    downsampling: int = 8

    iterations: int = 2


@dataclasses.dataclass
class PhaseRampRemovalOptions:
    iterations: int = 5

    downsampling: int = 8




