import dataclasses
from dataclasses import field
from numbers import Number
from typing import Optional, Sequence
from llama.api import enums
from llama.api.options.transform import CropOptions


@dataclasses.dataclass
class CITestOptions:
    test_data_name: str

    update_tester_results: bool = False

    atol: float = 1e-3

    rtol: float = 1e-3

    proj_idx: tuple[int] = (0,)

    save_temp_files: bool = False

    stop_on_error: bool = False
