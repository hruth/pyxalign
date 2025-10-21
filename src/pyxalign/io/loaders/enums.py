from enum import StrEnum, auto


class LoaderType(StrEnum):
    FOLD_SLICE_V1 = auto()
    FOLD_SLICE_V2 = auto()
    PEAR_V1 = auto()


class ExperimentInfoSourceType(StrEnum):
    LAMNI_DAT_FILE = auto()
    PTYCHO_FOLDERS = auto()
    BEAMLINE_2IDE_MDA_FILE = auto()


class ExperimentType(StrEnum):
    LYNX = auto()
    BEAMLINE_2IDE_PTYCHO = auto()
    BEAMLINE_2IDD_PTYCHO = auto()
    BEAMLINE_2IDE_XRF = auto()
