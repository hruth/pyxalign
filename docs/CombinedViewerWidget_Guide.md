# CombinedViewerWidget User Guide

## Overview

The `CombinedViewerWidget` is a unified interface that combines two powerful tools for laminography alignment:

1. **Projection Viewer** - Interactive visualization and manipulation of projections
2. **PMA Runner** - Projection Matching Alignment workflow interface

These tools are accessible through an icon-based sidebar navigation, allowing you to seamlessly switch between viewing projections and running alignment operations.

## Features

- **Unified Interface**: Single window for all alignment tasks
- **Sidebar Navigation**: Icon-based page selection for easy switching
- **Projection Viewing**: Full ProjectionViewer capabilities including:
  - Array visualization with multiple display modes
  - Mask creation and editing
  - Scan removal tools
  - Reconstruction parameter tuning
  - Applied shifts visualization
- **PMA Alignment**: Complete PMAMasterWidget functionality including:
  - Options configuration
  - Multi-resolution alignment sequencing
  - Real-time results visualization
  - Results collection and comparison

## Installation

The widget is part of the `pyxalign.interactions` module and is automatically available after installing pyxalign.

```python
from pyxalign.interactions import CombinedViewerWidget
```

## Basic Usage

### Creating the Widget

```python
from pyxalign.interactions import CombinedViewerWidget
from pyxalign.data_structures.task import load_task

# Load your alignment task
task = load_task("path/to/your/task.h5")

# Create the combined viewer
widget = CombinedViewerWidget(task)

# Show the widget
widget.show()
```

### Using the Example Script

A standalone example script is provided:

```bash
cd /home/beams0/HRUTH/code/lamino_development/pyxalign
python examples/combined_viewer_example.py /path/to/task.h5
```

## Widget Structure

### Input Parameters

The `CombinedViewerWidget` requires a single input:

- **task** (`LaminographyAlignmentTask`): The alignment task containing:
  - `phase_projections`: Used for the Projection Viewer
  - Full task object: Passed to PMA Runner for alignment

### Interface Layout

```
┌─────────────────────────────────────────────────────┐
│  [Icon] Projection Viewer  │                        │
│  [Icon] PMA Runner         │   Content Area         │
│                            │                        │
│                            │  (Shows selected page) │
│                            │                        │
│         Sidebar            │                        │
└─────────────────────────────────────────────────────┘
```

## Page Descriptions

### Page 1: Projection Viewer

The Projection Viewer page provides comprehensive projection visualization and manipulation tools:

#### Features:
- **Array Selection**: Switch between projections, masks, forward projections, and residuals
- **Image Display**: Interactive ArrayViewer with zoom, pan, and navigation
- **Mask Tools**:
  - Create masks manually
  - Generate masks from ROI
  - Edit existing masks
- **Scan Removal**: Remove problematic projections
- **Parameter Tuning**: Adjust reconstruction parameters (for phase projections)
- **Shifts Visualization**: View applied shift history

#### Typical Workflow:
1. View projections to identify issues
2. Create or adjust masks if needed
3. Remove bad scans
4. Tune reconstruction parameters
5. Review applied shifts

### Page 2: PMA Runner

The PMA Runner page provides the complete projection matching alignment workflow:

#### Features:
- **Options Configuration**: Edit projection matching parameters
- **Sequencer**: Set up multi-resolution alignment sequences
- **Real-time Visualization**: View alignment results as they're computed
- **Results Collection**: Compare multiple alignment runs
- **Progress Monitoring**: Track alignment sequence progress

#### Typical Workflow:
1. Configure alignment options
2. Set up multi-resolution sequence (if desired)
3. Start alignment
4. Monitor progress in real-time
5. Review and compare results

## Advanced Usage

### Customizing Icons

The widget supports custom icons. You can modify the `_get_icon()` method in `CombinedViewerWidget` to use custom icon files:

```python
def _get_icon(self, icon_name: str) -> QIcon:
    icon_paths = {
        "view": "/path/to/view_icon.png",
        "align": "/path/to/align_icon.png",
    }
    if icon_name in icon_paths:
        return QIcon(icon_paths[icon_name])
    else:
        return QIcon()
```

### Adding Additional Pages

You can extend the widget by adding more pages:

```python
widget = CombinedViewerWidget(task)

# Add a custom page
from PyQt5.QtWidgets import QLabel
custom_page = QLabel("Custom Analysis Tools")
widget.addPage(custom_page, "Custom", QIcon("custom_icon.png"))

widget.show()
```

### Accessing Sub-widgets

The sub-widgets are accessible as attributes:

```python
widget = CombinedViewerWidget(task)

# Access the projection viewer
projection_viewer = widget.projection_viewer

# Access the PMA widget
pma_widget = widget.pma_widget

# You can now interact with these widgets programmatically
# For example:
# projection_viewer.update_arrays()
# pma_widget.start_alignment_sequence()
```

## Requirements

- **Task with Phase Projections**: The widget works best when the task has phase projections unwrapped
- **Qt5**: PyQt5 must be installed and configured
- **GPU Support**: PMA operations may require GPU (CuPy)

## Error Handling

### No Phase Projections Available

If the task doesn't have phase projections, the Projection Viewer page will show a placeholder message:

> "No phase projections available. Please unwrap phase first."

**Solution**: Unwrap the phase before creating the widget:

```python
from pyxalign.data_structures.task import load_task

task = load_task("path/to/task.h5")

# Unwrap phase if not already done
if task.phase_projections is None:
    task.get_unwrapped_phase()

# Now create the widget
widget = CombinedViewerWidget(task)
widget.show()
```

## Tips and Best Practices

1. **Save your task frequently**: Use `task.save_task("path.h5")` to preserve your work

2. **Use multi-resolution sequences**: Start with coarse resolution for speed, then refine

3. **Review projections before alignment**: Use the Projection Viewer to identify and fix issues first

4. **Compare multiple alignments**: Use the Results Collection tab in PMA Runner to compare different parameter sets

5. **Monitor GPU memory**: Large datasets may require memory management

## Integration with Other Tools

The `CombinedViewerWidget` integrates seamlessly with other pyxalign tools:

```python
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.interactions import CombinedViewerWidget

# Create a task programmatically
task = LaminographyAlignmentTask(
    complex_projections=complex_projs,
    options=options
)

# Unwrap phase
task.get_unwrapped_phase()

# Launch the combined viewer
widget = CombinedViewerWidget(task)
widget.show()
```

## Keyboard Shortcuts

The widget inherits keyboard shortcuts from its sub-widgets:

- **ArrayViewer shortcuts** (in Projection Viewer page):
  - Arrow keys: Navigate through projections
  - Mouse wheel: Zoom in/out
  - Click+Drag: Pan image

- **Standard Qt shortcuts**:
  - Ctrl+Q: Quit application (if running standalone)

## Technical Details

### Class Hierarchy

```
QWidget
└── SidebarNavigator
    └── CombinedViewerWidget
```

### File Location

- **Source**: `pyxalign/interactions/combined_viewer.py`
- **Example**: `examples/combined_viewer_example.py`
- **Documentation**: `docs/CombinedViewerWidget_Guide.md`

### Dependencies

- PyQt5
- pyxalign.interactions.sidebar_navigator
- pyxalign.interactions.pma_runner
- pyxalign.interactions.viewers.arrays
- pyxalign.data_structures.task

## Troubleshooting

### Widget doesn't show up
- Ensure you're running in a Qt event loop
- Check that `widget.show()` is called
- Verify PyQt5 is properly installed

### Icons not displaying
- Icons use Qt standard icons by default
- Custom icons require valid file paths
- Text labels will always show even without icons

### Performance issues
- Large datasets may be slow to render
- Consider downsampling for preview
- Monitor GPU memory usage

## Examples

### Complete Example

```python
from PyQt5.QtWidgets import QApplication
import sys
from pyxalign.interactions import CombinedViewerWidget
from pyxalign.data_structures.task import load_task

def main():
    app = QApplication(sys.argv)

    # Load the task
    task = load_task("my_alignment_task.h5")

    # Ensure phase is unwrapped
    if task.phase_projections is None:
        print("Unwrapping phase...")
        task.get_unwrapped_phase()

    # Create and show the widget
    widget = CombinedViewerWidget(task)
    widget.show()

    # Run
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
```

## Support

For issues or questions:
- Check the inline documentation in the source code
- Review the example script
- Examine related widgets: `ProjectionViewer`, `PMAMasterWidget`

## Version History

- **v1.0**: Initial implementation with Projection Viewer and PMA Runner pages
