# Cline Rules for src/pyxalign/interactions/

These rules apply specifically to files in the `src/pyxalign/interactions/` directory and override global rules where specified.

## PyQt5 GUI Development Guidelines

### Widget Class Organization
- Organize widget classes with the following method order:
  1. `__init__` method
  2. Public interface methods
  3. Signal handlers (methods starting with `on_` or ending with `_handler`)
  4. Private helper methods (starting with `_`)
  5. Property getters/setters

### Signal and Slot Naming
- Signal handlers should be named `on_<signal_source>_<signal_name>` or `<action>_handler`
- Examples: `on_button_clicked`, `on_combo_box_changed`, `data_loaded_handler`
- Connect signals immediately after widget creation when possible

### Layout Management
- Use descriptive variable names for layouts: `main_layout`, `button_layout`, `form_layout`
- Group related widgets in logical layout sections
- Add comments for complex layout hierarchies

### Widget Initialization
- Create widgets in logical groups (e.g., all buttons together, all input fields together)
- Set widget properties immediately after creation
- Use `setSizePolicy` and styling consistently

## Documentation Requirements

### Module Docstrings
- All modules must have a module-level docstring explaining the module's purpose
- Include brief description of main classes/functions
- Mention any external dependencies or special requirements

Example:
```python
"""
Interactive mask threshold selector based on pyqtgraph and the shared
IndexSelectorWidget used elsewhere in pyxalign.

This module provides tools for interactively selecting binary thresholds
for automatically generated probe-patch masks.
"""
```

### Class Docstrings
- All widget classes must have docstrings
- Include purpose, main functionality, and any signals emitted
- Document constructor parameters

Example:
```python
class ThresholdSelector(QWidget):
    """
    Interactive tool to choose a binary-threshold for automatically
    generated "probe-patch" masks.

    Signals
    -------
    masks_created : np.ndarray
        Emitted once the user presses *Select and Finish*, containing the
        clipped/binary masks.
    """
```

### Method Docstrings
- Required for all public methods with complex logic
- Required for all signal handlers
- Use Google-style docstrings with Args, Returns, and Raises sections when applicable

## Import Guidelines (Interactions-Specific)

### PyQt5 Import Organization
```python
# Standard library
import sys
from typing import Optional, List

# Third-party
import numpy as np
import pyqtgraph as pg

# PyQt5 imports (grouped by module)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
)

# Local pyxalign imports
from pyxalign.interactions.base import BaseWidget
from pyxalign.plotting.interactive.base import IndexSelectorWidget
```

### Import Sorting Rules
- Sort PyQt5 widget imports alphabetically within the parentheses
- Keep related PyQt5 modules together (QtCore, QtWidgets, QtGui)
- Local imports should be sorted by module hierarchy

## Code Organization Patterns

### Complex Widget Classes
- Break large `__init__` methods into smaller setup methods:
  - `setup_ui()` for basic widget creation
  - `setup_layouts()` for layout management
  - `setup_connections()` for signal/slot connections
  - `setup_initial_state()` for initial widget states

### Data Handling
- Use descriptive names for data containers: `standard_data`, `projection_array`, `alignment_results_list`
- Validate data inputs in setter methods
- Emit signals when data state changes significantly

### Threading and Async Operations
- Use clear naming for threaded operations: `load_data_async`, `process_projections_threaded`
- Always handle thread completion and error cases
- Update UI elements only from the main thread

## Error Handling Patterns

### User Input Validation
```python
try:
    value = float(input_text)
    if value < 0:
        raise ValueError("Value must be non-negative")
except ValueError as e:
    self.show_error_message(f"Invalid input: {e}")
    return
```

### Resource Management
- Always clean up widgets with `deleteLater()` when removing from layouts
- Use `sip.delete()` for complex widget hierarchies when needed
- Check for `sip.isdeleted()` before accessing widgets that may have been cleaned up

## Performance Guidelines

### Large Data Handling
- Use `pin_memory()` for GPU data transfers
- Implement lazy loading for large datasets
- Show progress indicators for operations > 1 second

### UI Responsiveness
- Use `QTimer` for periodic updates instead of tight loops
- Implement proper cancellation for long-running operations
- Update progress bars and status messages regularly

## Specific Widget Patterns

### Options Editors
- Use `BasicOptionsEditor` for dataclass editing
- Implement proper field validation and error handling
- Provide clear visual feedback for invalid inputs

### Sequencer Widgets
- `SequencerWidget` manages multiple `SequencerItem` instances for creating option sequences
- `SequencerItem` provides nested dataclass exploration with categorized combo boxes
- Both widgets support basic/advanced field categorization using `basic_options_list`

#### SequencerItem Implementation Details
- **Field Categorization**: Use `_get_categorized_fields()` to separate basic/advanced fields based on full dotted paths
- **Path Context Tracking**: Use `_build_current_path_prefix()` to maintain nested path context for proper categorization
- **Combo Box Population**: Create categorized sections with bold headers: `<b>---BASIC---</b>` and `<b>---ADVANCED---</b>`
- **Section Header Handling**: Make section headers non-selectable in `on_combo_box_changed()`
- **Nested Support**: Categorization works at all nesting levels (e.g., "downsample.scale" categorized under downsample combo box)

#### SequencerWidget Integration
- Pass `basic_options_list` parameter to constructor and propagate to all SequencerItem instances
- Ensure all SequencerItem creation methods (`add_new_sequencer`, `duplicate_last_sequence`) include categorization settings
- Maintain consistency with `BasicOptionsEditor` by using the same basic options list (e.g., `basic_pma_settings`)

#### Field Categorization Logic
```python
# Example basic options list for PMA settings
basic_pma_settings = [
    "high_pass_filter",    # Top-level basic field
    "iterations",          # Top-level basic field
    "keep_on_gpu",        # Top-level basic field
    "downsample",         # Parent dataclass (basic)
    "downsample.scale",   # Nested field (basic)
]

# Categorization uses full dotted paths:
# - "downsample" appears in basic section of top-level combo
# - "scale" appears in basic section of downsample combo
# - "enabled" appears in advanced section of downsample combo
```

### Data Viewers
- Implement consistent navigation patterns (play/pause/step)
- Use `IndexSelectorWidget` for array navigation
- Provide multiple view modes when appropriate (amplitude/phase for complex data)

### File Dialogs
- Use `CustomFileDialog` pattern for file/folder selection
- Validate file paths immediately after selection
- Provide clear error messages for invalid paths

## Testing Considerations

### Widget Testing
- Ensure widgets can be created without external dependencies
- Provide mock data for testing complex interactions
- Test signal emission and handling

### Integration Testing
- When real data testing is needed, ask the maintainer to provide the appropriate data loading code
- The maintainer will typically handle integration testing with real datasets
- Focus on unit testing and mock data scenarios unless specifically requested to test with real data
