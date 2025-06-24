from typing import Union
from pyxalign.io.loaders.lamni.lamni_loader_2 import LamniLoaderVersion2
from pyxalign.io.loaders.lamni.lamni_loader_1 import LamniLoaderVersion1
from pyxalign.io.loaders.lamni.options import (
    BaseLoadOptions,
    Beamline2IDELoadOptions,
    LYNXLoadOptions,
)
from pyxalign.io.loaders.lamni.pear_loader_1 import PearLoaderVersion1
from pyxalign.io.loaders.enums import ExperimentType, LoaderType
from pyxalign.io.loaders.xrf.options import XRFLoadOptions
from pyxalign.io.utils import OptionsClass

LoaderInstanceType = Union[LamniLoaderVersion1, LamniLoaderVersion2, PearLoaderVersion1]
LoaderClassType = Union[
    type[LamniLoaderVersion1], type[LamniLoaderVersion2], type[PearLoaderVersion1]
]


def get_loader_class_by_enum(key: LoaderType) -> LoaderClassType:
    return {
        LoaderType.LAMNI_V1: LamniLoaderVersion1,
        LoaderType.LAMNI_V2: LamniLoaderVersion2,
        LoaderType.PEAR_V1: PearLoaderVersion1,
    }[key]


def get_loader_options_by_enum(key: ExperimentType) -> OptionsClass:
    return {
        ExperimentType.LYNX: LYNXLoadOptions(
            dat_file_path=None,
            base=BaseLoadOptions(parent_projections_folder=None),
        ),
        ExperimentType.BEAMLINE_2IDE_PTYCHO: Beamline2IDELoadOptions(
            mda_folder=None,
            base=BaseLoadOptions(parent_projections_folder=None),
        ),
        ExperimentType.BEAMLINE_2IDE_XRF: XRFLoadOptions(),
    }[key]


def get_experiment_type_enum_from_options(options: OptionsClass) -> ExperimentType:
    if isinstance(options, LYNXLoadOptions):
        return ExperimentType.LYNX
    elif isinstance(options, Beamline2IDELoadOptions):
        return ExperimentType.BEAMLINE_2IDE_PTYCHO
    elif isinstance(options, XRFLoadOptions):
        return ExperimentType.BEAMLINE_2IDE_XRF
