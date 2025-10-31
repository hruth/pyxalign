"""
File loading and preparation module.

This module provides the options and functions for loading the raw
data into pyxalign's standard input format format.
"""

from . import pear, xrf
from .base import StandardData
from .utils import convert_projection_dict_to_array

__all__ = [
    "pear",
    "xrf",
    "StandardData",
    "convert_projection_dict_to_array",
]
