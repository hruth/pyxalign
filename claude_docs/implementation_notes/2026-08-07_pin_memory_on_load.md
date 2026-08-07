# Pin Memory on Load

**Date:** 2026-08-07

## Problem

When loading a task from a checkpoint, projection arrays and masks are loaded into ordinary (pageable) host memory. For GPU-heavy workflows this means the CUDA runtime must page-lock the arrays on-the-fly every time they are transferred to the device, adding latency and overhead.

## Solution

Added a `pin_memory` option to the task/projection loading path so that arrays can be placed directly into CUDA pinned (page-locked) host memory at load time, eliminating the per-transfer pinning cost.

## Changes

### `src/pyxalign/io/load.py`
- `load_ptycho_projections()` — added `pin_memory: bool = False` parameter; passes it to `load_projections_object()`.
- `load_projections_object()` — added `pin_memory: bool = False` parameter. When `True`, calls `gpu_utils.pin_memory()` on the projection data array and, if present, on the masks array before constructing the `Projections` object.

### `src/pyxalign/data_structures/task.py`
- `load_task()` — added `pin_memory: bool = False` parameter; passes it to `load_ptycho_projections()`.

### `src/pyxalign/autorunner/config.py`
- `CheckpointConfig` — added `pin_memory_on_load: bool = False` field with a descriptive docstring.

### `src/pyxalign/autorunner/abstract.py`
- `handle_checkpoint` decorator — both `load_task_wrapped()` calls (custom task path and default checkpoint path) now pass `pin_memory=self.config.checkpoint.pin_memory_on_load`.

## Autorunner UI

`pin_memory_on_load` is automatically included in the autorunner configuration window via the existing `_get_high_level_config_options()` logic, which adds all `CheckpointConfig` fields to the basic options list under the `checkpoint.*` namespace.

## Usage

**Programmatic:**
```python
task = load_task("checkpoint.h5", pin_memory=True)
```

**Autorunner:**
Set `checkpoint.pin_memory_on_load = True` in the autorunner configuration window (or in the state YAML file).
