# Fill Missing Cone GUI and snake_case Refactor

**Date:** 2026-08-25

## Changes

### 1. `missing_cone.py` — snake_case conversion

All camelCase identifiers in this module were renamed to snake_case. This is a **breaking API change** — no backward-compatible aliases were added.

Key renames:
| Old name | New name |
|---|---|
| `fill_missing_cone(laminoAngle, deltaBackground, ...)` | `fill_missing_cone(lamino_angle, delta_background, ...)` |
| `applyLaminoConstraints` | `apply_lamino_constraints` |
| `getLaminoFourierMask` | `get_lamino_fourier_mask` |
| `getMask` | `get_mask` |
| `interpFT_3D` | `interp_ft_3d` |
| `cropPad3D` | `crop_pad_3d` |
| `blockProc` | `block_proc` |
| `padInputs` | `pad_inputs` |

The decorator kwargs `"blockSize"` and `"borderSize"` were also renamed to `"block_size"` and `"border_size"` consistently throughout.

**Note:** `keep_on_gpu` in `get_lamino_fourier_mask` was present in the original but never used; it was retained in the signature to avoid altering the positional call sites.

### 2. `api/options/missing_cone.py` — new file

Defines `FillMissingConeOptions(BaseOptions)` with fields:
- `delta_background: float = 0.02`
- `delta_maximal: float = 0.4`
- `mask_relax: float = 0.05`
- `max_scale: int = 16`
- `n_iter: int = 10`
- `tv_lambda: float = 1e-7`
- `crop_3d: Crop3DOptions` — crop applied to the volume before `fill_missing_cone` is called.

Exported from `api/options/__init__.py`.

### 3. `interactions/missing_cone_window.py` — new GUI

**`FillMissingConeWindow(QWidget)`** — two-tab window.

**Run tab:**
- Left panel: laminography angle spinbox, "Select 3D Crop" button (opens `GetCrop3DOptionsFromSelector`), `BasicOptionsEditor` for `FillMissingConeOptions` (skipping `crop_3d`), green "Run" button.
- Right panel: `_VolumeViewerPanel` — orthogonal three-view layout (depth/side1/side2) with synchronized color limits and `include_array_saving_widget=True` on the depth viewer.

**History tab:**
- Left panel: table recording each run (run #, timestamp, lamino angle, all options fields, crop summary) and a red "Delete Volume for Selected Entry" button.
- Right panel: `_VolumeViewerPanel` that updates when a table row is selected.

Clicking "Delete Volume" frees the stored `np.ndarray` from the history entry to save memory, while keeping the table row (text grayed out to indicate the volume is gone).

**`_VolumeViewerPanel(QWidget)`** — internal helper. Mirrors `VolumeViewer` layout but exposes array saving on the depth view. Supports lazy initialization (shows placeholder label until a volume is set via `update_volume()`).

### 4. `gui/__init__.py`

Added `launch_fill_missing_cone_gui` to the public GUI facade.

Usage example:
```python
import pyxalign

options = pyxalign.api.options.FillMissingConeOptions(n_iter=5)
gui = pyxalign.gui.launch_fill_missing_cone_gui(
    volume=my_3d_volume,
    lamino_angle=79.0,
    options=options,
)
```
