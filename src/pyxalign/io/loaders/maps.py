from typing import Union
from pyxalign.io.loaders.pear.fold_slice_loader_2 import FoldSliceLoaderVersion2
from pyxalign.io.loaders.pear.fold_slice_loader_1 import FoldSliceLoaderVersion1
from pyxalign.io.loaders.pear.options import (
    BaseLoadOptions,
    Beamline2IDDLoadOptions,
    Beamline2IDELoadOptions,
    LYNXLoadOptions,
    LoaderType,
)
from pyxalign.io.loaders.pear.pear_loader_1 import PearLoaderVersion1
from pyxalign.io.loaders.enums import ExperimentType
from pyxalign.io.loaders.xrf.options import Beamline2IDEXRFLoadOptions
from pyxalign.io.utils import OptionsClass

LoaderInstanceType = Union[FoldSliceLoaderVersion1, FoldSliceLoaderVersion2, PearLoaderVersion1]
LoaderClassType = Union[
    type[FoldSliceLoaderVersion1], type[FoldSliceLoaderVersion2], type[PearLoaderVersion1]
]


def get_loader_class_by_enum(key: LoaderType) -> LoaderClassType:
    return {
        LoaderType.FOLD_SLICE_V1: FoldSliceLoaderVersion1,
        LoaderType.FOLD_SLICE_V2: FoldSliceLoaderVersion2,
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
        ExperimentType.BEAMLINE_2IDD_PTYCHO: Beamline2IDDLoadOptions(
            mda_folder=None,
            base=BaseLoadOptions(parent_projections_folder=None),
        ),
        ExperimentType.BEAMLINE_2IDE_XRF: Beamline2IDEXRFLoadOptions(),
    }[key]


def get_experiment_type_enum_from_options(options: OptionsClass) -> ExperimentType:
    if isinstance(options, LYNXLoadOptions):
        return ExperimentType.LYNX
    elif isinstance(options, Beamline2IDELoadOptions):
        return ExperimentType.BEAMLINE_2IDE_PTYCHO
    elif isinstance(options, Beamline2IDDLoadOptions):
        return ExperimentType.BEAMLINE_2IDD_PTYCHO
    elif isinstance(options, Beamline2IDEXRFLoadOptions):
        return ExperimentType.BEAMLINE_2IDE_XRF
