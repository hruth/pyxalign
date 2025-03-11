from typing import Union
from llama.io.loaders.lamni.lamni_loader_2 import LamniLoaderVersion2
from llama.io.loaders.lamni.lamni_loader_1 import LamniLoaderVersion1
from llama.io.loaders.lamni.lamni_loader_3 import LamniLoaderVersion3
from llama.io.loaders.enums import LoaderType

LoaderInstanceType = Union[LamniLoaderVersion1, LamniLoaderVersion2]
LoaderClassType = Union[type[LamniLoaderVersion1], type[LamniLoaderVersion2]]


def get_loader_class_by_enum(key: LoaderType) -> LoaderClassType:
    return {
        LoaderType.LAMNI_V1: LamniLoaderVersion1,
        LoaderType.LAMNI_V2: LamniLoaderVersion2,
        LoaderType.LAMNI_V3: LamniLoaderVersion3,
    }[key]