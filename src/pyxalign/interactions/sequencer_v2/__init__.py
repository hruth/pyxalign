"""
Version 2 of the sequencer widget - block-based alignment sequence management.

This version organizes alignment runs into discrete blocks, where each block
represents a single alignment execution with potentially multiple parameter changes.
This makes the relationship between parameters and alignment runs much clearer.
"""

from .sequencer import SequencerWidgetV2
from .sequencer_item import SequencerItemV2, ParameterRow

__all__ = ["SequencerWidgetV2", "SequencerItemV2", "ParameterRow"]
