from typing import Union
from pyxalign.io.loaders.pear.fold_slice_loader_2 import FoldSliceLoaderVersion2
from pyxalign.io.loaders.pear.fold_slice_loader_1 import FoldSliceLoaderVersion1

import pyxalign.io.loaders.pear.options as pear_options
import pyxalign.io.loaders.xrf.options as xrf_options

from pyxalign.io.loaders.pear.pear_loader_1 import PearLoaderVersion1
from pyxalign.io.loaders.enums import ExperimentType
from pyxalign.api.types import OptionsClass

LoaderClassType = Union[
    type[FoldSliceLoaderVersion1], type[FoldSliceLoaderVersion2], type[PearLoaderVersion1]
]

def get_loader_class_by_enum(key: pear_options.LoaderType) -> LoaderClassType:
    return {
        pear_options.LoaderType.FOLD_SLICE_V1: FoldSliceLoaderVersion1,
        pear_options.LoaderType.FOLD_SLICE_V2: FoldSliceLoaderVersion2,
        pear_options.LoaderType.PEAR_V1: PearLoaderVersion1,
    }[key]


def get_loader_options_by_enum(key: ExperimentType) -> OptionsClass:
    return {
        ExperimentType.LYNX: pear_options.LYNXLoadOptions(
            dat_file_path=None,
            base=pear_options.BaseLoadOptions(parent_projections_folder=""),
        ),
        ExperimentType.BEAMLINE_2IDE_PTYCHO: pear_options.Microprobe2IDELoadOptions(
            mda_folder=None,
            base=pear_options.BaseLoadOptions(parent_projections_folder=""),
        ),
        ExperimentType.BEAMLINE_2IDD_PTYCHO: pear_options.BNP2IDDLoadOptions(
            mda_folder=None,
            base=pear_options.BaseLoadOptions(parent_projections_folder=""),
        ),
        ExperimentType.BEAMLINE_2IDE_XRF: xrf_options.XRF2IDELoadOptions(),
        ExperimentType.BEAMLINE_12IDE_PTYCHO: pear_options.Ptycho12IDELoadOptions(),
    }[key]


def get_experiment_type_enum_from_options(options: OptionsClass) -> ExperimentType:
    if isinstance(options, pear_options.LYNXLoadOptions):
        return ExperimentType.LYNX
    elif isinstance(options, pear_options.Microprobe2IDELoadOptions):
        return ExperimentType.BEAMLINE_2IDE_PTYCHO
    elif isinstance(options, pear_options.BNP2IDDLoadOptions):
        return ExperimentType.BEAMLINE_2IDD_PTYCHO
    elif isinstance(options, xrf_options.XRF2IDELoadOptions):
        return ExperimentType.BEAMLINE_2IDE_XRF
    elif isinstance(options, pear_options.Ptycho12IDELoadOptions):
        return ExperimentType.BEAMLINE_12IDE_PTYCHO

    # # above part doesn't run right when using reloading features during development
    # # other users pls ignore
    # if options.__class__.__qualname__ == pear_options.LYNXLoadOptions.__qualname__:
    #     return ExperimentType.LYNX
    # elif (
    #     options.__class__.__qualname__
    #     == pear_options.Microprobe2IDELoadOptions.__qualname__
    # ):
    #     return ExperimentType.BEAMLINE_2IDE_PTYCHO
    # elif options.__class__.__qualname__ == pear_options.BNP2IDDLoadOptions.__qualname__:
    #     return ExperimentType.BEAMLINE_2IDD_PTYCHO
    # elif options.__class__.__qualname__ == xrf_options.XRF2IDELoadOptions.__qualname__:
    #     return ExperimentType.BEAMLINE_2IDE_XRF
