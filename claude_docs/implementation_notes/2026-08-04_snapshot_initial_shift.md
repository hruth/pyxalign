# Snapshot-based Initial Shift for PMA Alignment

**Date:** 2026-08-04

## Problem

The "Initial Shift" dropdown on the Defaults side of the Configure & Start window only offered entries derived from the current session's alignment results ("Result 0", "Result 1", etc.) and "Previous"/"None". There was no way to initialize the next alignment from a specific snapshot in the PMA history viewer without first staging and applying that snapshot's shift.

## Solution

Added a new button — **"Initialize next alignment with selected snapshot shift"** — in `PMASequenceViewer`, placed above the existing "Stage && Apply Selected Snapshot Shift" button. Clicking it promotes the selected snapshot into the "Initial Shift" dropdown of the connected `PMAMasterWidget` as **"Snapshot N"** (where N is the snapshot's index in the sequence table) and immediately selects it.

### Files changed

- **`src/pyxalign/interactions/viewers/pma_tracking.py`**
  - `PMASequenceViewer.__init__`: added `on_initialize_with_snapshot: Optional[Callable[[int], None]]` parameter.
  - `_build_ui`: added `self.initialize_with_snapshot_button` above the stage button; starts disabled until a snapshot with a `final_shift` is selected.
  - `_update_stage_button_state`: extended to also enable/disable `initialize_with_snapshot_button`.
  - `_on_initialize_with_snapshot_clicked`: new handler that calls the callback with the current snapshot index.

- **`src/pyxalign/interactions/pma_runner.py`**
  - `PMAMasterWidget.__init__`: added `self._snapshot_initial_shift_indices: list[int]` to persist promoted snapshot indices across combobox rebuilds.
  - `add_snapshot_initial_shift(snapshot_index)`: new method — adds "Snapshot N" to the combobox if absent, then selects it.
  - `update_initial_shift_combobox`: re-adds snapshot entries after clearing and rebuilding.
  - `get_initial_shift`: handles the `"Snapshot N"` prefix by looking up `task.pma_sequence.snapshots[N]` directly.
  - `on_open_pma_sequence_viewer_clicked`: passes `on_initialize_with_snapshot=self.add_snapshot_initial_shift` to `PMASequenceViewer`.

## Behavior

- The new button is **disabled by default** until the user selects a snapshot with a completed `final_shift`.
- Clicking the button does **not** stage or apply any shift — it only adds the entry to the dropdown and selects it for the next run.
- The "Snapshot N" entry survives combobox rebuilds (e.g., after new alignment results arrive) because `_snapshot_initial_shift_indices` tracks which snapshots have been promoted.
- Multiple snapshots can be promoted; each appears as a separate entry. Promoting the same snapshot twice is a no-op (entry already present, selection still switches to it).
- The `PMASequenceViewer` opened without a connected runner (e.g., standalone) shows the button permanently disabled with an appropriate tooltip.
