from typing import Optional
import h5py
import numpy as np
import os
from abc import ABC
import re
from pyxalign.io.loaders.utils import (
    border,
    generate_input_user_prompt,
    get_boolean_user_input,
    load_h5_group,
    parallel_load_all_projections,
)
from pyxalign.timing.timer_utils import InlineTimer, timer


class XRFLoader(ABC):
    pass