# PMA Sequence Volume: Save/Load and Delete Button

**Date:** 2026-08-04

## Problem

1. The "Save Current Task" button in the autorunner GUI was calling `task.save_task()` without `save_pma_sequence_volumes=True`, so PMA sequence volumes were silently dropped whenever users manually saved a task from the GUI.
2. The `AutorunnerConfig` had no way to control whether PMA sequence volumes were loaded when resuming from a checkpoint.
3. The PMA Sequence Viewer had no way to delete a volume array from a snapshot in-session (e.g. to free memory after inspecting it).

## Changes

### `src/pyxalign/interactions/autorunner/wrapper.py`
- `AutorunnerGUIWrapper._on_save_task`: changed `self.task.save_task(file_path)` to `self.task.save_task(file_path, save_pma_sequence_volumes=True)` so volumes are included when the user explicitly saves a task.

### `src/pyxalign/autorunner/config.py`
- Added `load_pma_sequence_volumes: bool = True` to `AutorunnerConfig`. When `True` (the default), checkpoint loads include PMA sequence volumes.

### `src/pyxalign/autorunner/abstract.py`
- `handle_checkpoint`: both `load_task` calls (for custom path and default checkpoint path) now pass `load_pma_sequence_volumes=self.config.load_pma_sequence_volumes`.

### `src/pyxalign/interactions/viewers/pma_tracking.py`
- Added a red "Delete Volume Array" button to the top bar of `PMASequenceViewer`. It is only visible when the "volume" display type is selected.
- The button prompts for confirmation, then sets `snapshot.volume = None` on the selected snapshot and refreshes the viewer.
- `_sync_control_visibility` updated to show/hide the new button.
