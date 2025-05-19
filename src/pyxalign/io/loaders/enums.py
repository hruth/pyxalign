from enum import StrEnum, auto


class LamniLoaderType(StrEnum):
    LAMNI_V1 = auto()
    LAMNI_V2 = auto()
    LAMNI_V3 = auto()


class ExperimentInfoSourceType(StrEnum):
    LAMNI_DAT_FILE = auto()
    PTYCHO_FOLDERS = auto()
    BEAMLINE_2IDE_MDA_FILE = auto()
