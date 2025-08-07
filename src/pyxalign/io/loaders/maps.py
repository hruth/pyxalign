from typing import Union
from pyxalign.io.loaders.pear.fold_slice_loader_2 import FoldSliceLoaderVersion2
from pyxalign.io.loaders.pear.fold_slice_loader_1 import FoldSliceLoaderVersion1
from pyxalign.io.loaders.pear.pear_loader_1 import PearLoaderVersion1
from pyxalign.io.loaders.enums import LoaderType

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
