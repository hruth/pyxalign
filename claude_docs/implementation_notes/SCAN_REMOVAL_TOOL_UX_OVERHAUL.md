# ScanRemovalTool UX Overhaul

**Date:** 2026-08-03
**File:** `src/pyxalign/interactions/viewers/arrays.py`

## Problem Solved

The original `ScanRemovalTool` only supported removing scans one at a time via a checkbox tied to the array viewer's slider. There was no way to remove large numbers of scans efficiently.

## Solution

Added range-based removal alongside the existing individual scan mode, controlled by a selection mode dropdown. Overhauled the button/checkbox interaction model and improved layout consistency.

## Changes

### Selection Mode Dropdown

A `QComboBox` at the top of the widget with three options (stored as class constants):
- `SELECTION_MODE_INDIVIDUAL` — original per-scan behavior
- `SELECTION_MODE_SCAN_NUMBER_RANGE` — specify a start/end scan number
- `SELECTION_MODE_ANGLE_RANGE` — specify a start/end angle (degrees)

Mode changes are handled by `_on_selection_mode_changed`, which toggles panel and table visibility and clears the range inputs.

### Range Mode UI

A panel (`range_mode_widget`) with a `QFormLayout` containing "Start" and "End" `QLineEdit` fields. Inputs are cleared whenever the user switches to a range mode. Staged ranges are stored in `scan_ranges_table` (columns: Start, End, Range Type where type is `"scan number"` or `"angle"`).

### Add / Remove Staged Buttons

Replaced the "Mark for removal" checkbox with two light-blue (`#ADD8E6`) `QPushButton`s in a horizontal row:

- **"Add to scans staged for removal"** — always enabled; dispatches to `_add_individual_scan_to_staged` or `_stage_range_for_removal` based on mode
- **"Remove from scans staged for removal"** — disabled until one or more rows are selected in the active staging table; removes all selected rows (highest index first to avoid index shifting)

Both staging tables use `QTableWidget.SelectRows` selection behavior. Their `itemSelectionChanged` signals drive `_update_remove_button_state`.

### Duplicate Prevention

- Individual mode: `_add_individual_scan_to_staged` checks the scan number column before inserting; silently no-ops on duplicates
- Range mode: `_stage_range_for_removal` checks all three fields (start, end, range type) before inserting; silently no-ops on duplicates

### Permanent Removal

`remove_staged_projections` collects scan numbers from both tables:
1. Direct scan numbers from `staged_for_removal_table`
2. Ranges resolved via `_get_scan_numbers_from_ranges`, which looks up each staged range against `projections.scan_numbers` / `projections.angles` with inclusive `<=` comparisons

Results are deduplicated before calling `projection_drop_function`.

### Layout Changes

- "Permanently Remove Scans" button moved **above** the "Previously removed scans" table and styled red (`#CC3333`, white bold text)
- `QFrame(HLine/Sunken)` separator placed between the button and the previously-removed section to visually divide interactive from read-only areas
- `individual_mode_widget` and `range_mode_widget` given `QSizePolicy.Fixed` vertical policy to prevent vertical stretching when the window is resized

## API Impact

No public API changes. `ScanRemovalTool.__init__` signature is unchanged. The `projections_removed` signal still fires after successful removal.

## Removed

- `mark_for_removal_check_box` (`QCheckBox`)
- `update_mark_for_removal_check_box` method
- `update_staged_for_removal_list` method
- `_remove_selected_range` method
- Slider → `update_mark_for_removal_check_box` signal connection
