# Sequencer V2 Implementation Summary

## Overview

This document summarizes the changes made to create a new version of the SequencerWidget that better groups parameters into cohesive alignment run blocks.

## Problem with V1

In the original sequencer (v1), each parameter change was represented as a separate item in a list, with a checkbox "Run alignment after value change" determining when an alignment run would occur. This made it unclear which parameters were grouped together for a single alignment run.

Example from v1:
```
[high_pass_filter: 0.0123] ☑ Run alignment after value change
[downsample: selected] ☐ Run alignment after value change
[scale: 16] ☑ Run alignment after value change
[high_pass_filter: 0.0456] ☑ Run alignment after value change
```

In this example, it's not immediately clear that `downsample.scale = 16` is part of the same alignment run as the previous checkbox.

## Solution with V2

In v2, alignment runs are organized into distinct visual blocks. Each block represents one call to `get_projection_matching_shift()` and can contain multiple parameter changes.

Example from v2:
```
┌─────────────────────────────────────────┐
│ Alignment Run Block 1                   │
│ - high_pass_filter: 0.0123              │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Alignment Run Block 2                   │
│ - downsample: selected                  │
│ - downsample.scale: 16                  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Alignment Run Block 3                   │
│ - high_pass_filter: 0.0456              │
└─────────────────────────────────────────┘
```

This makes it crystal clear which parameters are applied together in each alignment run.

## File Structure Changes

### Refactored to v1 folder
- `src/pyxalign/interactions/sequencer.py` → `src/pyxalign/interactions/sequencer_v1/sequencer.py`
- `src/pyxalign/interactions/sequencer_item.py` → `src/pyxalign/interactions/sequencer_v1/sequencer_item.py`
- Created `src/pyxalign/interactions/sequencer_v1/__init__.py`

### New v2 implementation
- `src/pyxalign/interactions/sequencer_v2/sequencer.py` - Main sequencer widget
- `src/pyxalign/interactions/sequencer_v2/sequencer_item.py` - Individual block and parameter row widgets
- `src/pyxalign/interactions/sequencer_v2/__init__.py` - Package initialization

### Updated files
- `src/pyxalign/interactions/pma_runner.py` - Updated to use `SequencerWidgetV2`

## Key Components

### SequencerWidgetV2
Main widget that manages multiple alignment run blocks. Provides:
- Add/remove/duplicate blocks
- Generate options sequence (one per block)
- Track changed settings per block
- Load/save sequence configurations

### SequencerItemV2
Represents a single alignment run block. Features:
- Visual grouping with a styled frame
- Multiple parameter rows within the block
- Add/remove parameter rows
- Block-level operations (insert, duplicate, remove)

### ParameterRow
Individual parameter selection row within a block. Provides:
- Nested dropdown menus for option selection
- Value editor for the selected option
- Remove button for this parameter

## API Compatibility

The v2 sequencer maintains the same API as v1:

```python
# Initialization
sequencer = SequencerWidgetV2(
    options,
    list_of_updated_settings,  # Optional
    basic_options_list,        # Optional
    parent,                    # Optional
)

# Generate options sequence
options_sequence = sequencer.generate_options_sequence(base_options)

# Get changed settings
changed_settings = sequencer.get_changed_settings_sequence()
```

The `list_of_updated_settings` format is the same - a list of dictionaries where each dictionary represents the parameter changes for one alignment run.

## Benefits

1. **Clearer visual organization**: Each alignment run is a distinct, visually grouped block
2. **Better understanding**: Users can immediately see which parameters are applied together
3. **Easier editing**: Add/remove parameters within a block without affecting other blocks
4. **More intuitive**: No more confusion about checkbox placement determining run boundaries
5. **Flexible**: Can have any number of parameters per block (1 to many)

## Migration Path

The old sequencer (v1) is preserved in `sequencer_v1/` and can still be imported if needed:

```python
from pyxalign.interactions.sequencer_v1 import SequencerWidget as SequencerWidgetV1
```

The PMAMasterWidget has been updated to use v2 by default, but can be easily switched back if issues arise.

## Testing

A test script has been created at `test_sequencer_v2.py` to verify basic functionality:
- Widget creation
- Options sequence generation
- Changed settings tracking
- Loading from list of dicts

## Future Enhancements

Potential future improvements:
- Drag-and-drop reordering of blocks
- Copy/paste parameters between blocks
- Block templates for common parameter combinations
- Visual diff showing changes between blocks
- Export/import sequence configurations
