from enum import StrEnum, auto


class LoaderType(StrEnum):
    LAMNI_V1 = auto()
    LAMNI_V2 = auto()
    PEAR_V1 = auto()


class ExperimentInfoSourceType(StrEnum):
    LAMNI_DAT_FILE = auto()
    PTYCHO_FOLDERS = auto()
    BEAMLINE_2IDE_MDA_FILE = auto()
