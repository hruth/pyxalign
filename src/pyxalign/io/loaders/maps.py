from typing import Union
from pyxalign.io.loaders.pear.lamni_loader_2 import LamniLoaderVersion2
from pyxalign.io.loaders.pear.lamni_loader_1 import LamniLoaderVersion1
from pyxalign.io.loaders.pear.pear_loader_1 import PearLoaderVersion1
from pyxalign.io.loaders.enums import LoaderType

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
