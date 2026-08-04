# Applied Shifts Tab: Switch to PyQtGraph + X-axis Toggle

**Date:** 2026-08-04

## Problem

The "Applied Shifts" tab in `AllShiftsViewer` used a matplotlib `FigureCanvas` with a `NavigationToolbar` for displaying horizontal and vertical shifts. The x-axis was always angle (deg) with no way to switch to scan number.

## Solution

Replaced the matplotlib panel in `AllShiftsViewer` (`src/pyxalign/interactions/viewers/arrays.py`) with a `pg.GraphicsLayoutWidget` containing two linked `PlotItem`s, and added two radio buttons below the checkbox group to let the user select the x-axis quantity.

### Changes

- Added `import pyqtgraph as pg` at module level (line ~63).
- `__init__`: replaced `self.sort_idx` with `self.scan_numbers = projections.scan_numbers`.
- `init_ui()`:
  - Removed matplotlib `Figure`, `FigureCanvas`, and `NavigationToolbar`.
  - Added a `pg.GraphicsLayoutWidget` with `plot_horizontal` (row 0) and `plot_vertical` (row 1); x-axes linked via `setXLink`.
  - Added an "X-axis" `QGroupBox` containing "Angle" and "Scan number" `QRadioButton`s. The `angle_radio.toggled` signal calls `update_plot`.
- Added `_get_x_axis_data()` helper: returns `(x_values, sort_idx, x_label)` based on the active radio button, sorting by angle or scan number accordingly.
- `update_plot()`: calls `plot.clear()` + `plot.addLegend()` on both plots each refresh, then iterates checked checkboxes and calls `plot.plot(x, y, pen, name)`.
- `refresh_data()`: now reloads `self.scan_numbers` alongside angles; removed old `self.sort_idx` assignment.

## Notes

- matplotlib imports are retained — they are still used by `ApplySavedAlignmentShiftDialog` and the `@switch_to_matplotlib_qt_backend` decorator elsewhere in the file.
- Colors come from `color_list` (matplotlib XKCD hex strings), which pyqtgraph's `mkPen` accepts.
- The pattern for `clear()` → `addLegend()` → `plot()` follows the convention already used in `pma_tracking.py`.
