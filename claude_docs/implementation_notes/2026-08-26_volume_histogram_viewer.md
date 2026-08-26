# VolumeHistogramViewer — 2026-08-26

## Problem

No existing GUI allowed quick, interactive inspection of the pixel-value
distribution within a sub-region of a reconstruction volume.

## Solution

New module: `src/pyxalign/interactions/histogram_viewer.py`

### `VolumeHistogramViewer(QWidget)`

Single-window GUI combining:

1. **ArrayViewer** (`hide_axis_controls=False`) — displays the 3D volume with
   axis-cycling controls so any slice plane can be inspected.
2. **Interactive RectROI overlay** — a red draggable/resizable rectangle drawn
   on top of the ArrayViewer, with four corner scale handles.
3. **Crop Region spinboxes** — four QSpinBoxes (Horizontal Offset, Vertical
   Offset, Horizontal Range, Vertical Range) that stay bidirectionally
   synchronised with the ROI.  Offsets are measured from the image centre,
   consistent with `ROISelector` in `roi_selector.py`.
4. **Calculate Histogram button** — calls
   `pg.RectROI.getArrayRegion()` to extract the crop, then computes a
   histogram with `np.histogram` and renders it as `pg.BarGraphItem`.

### Layout

```
QVBoxLayout
├── ArrayViewer (stretch=3)
├── QGroupBox "Crop Region"
│   └── QGridLayout: 4 × (label + QSpinBox)
├── QPushButton "Calculate Histogram"  (aligned right)
└── pg.PlotWidget (stretch=2)
```

### Coordinate system

Follows the same convention as `ROISelector`:
- The ROI position/size is stored in pyqtgraph image coordinates (x, y).
- Spinbox offsets are relative to the image centre (rounded down to integer).
- `_current_image_dimensions()` derives (pg_width, pg_height) from the volume
  shape and current slider axis (using the same transpose logic as
  `ArrayViewer.display_frame`).

## API changes

- New function `launch_volume_histogram_viewer` exported from
  `src/pyxalign/gui/__init__.py`.

## Usage

```python
import pyxalign.gui

# blocking
gui = pyxalign.gui.launch_volume_histogram_viewer(volume, wait_until_closed=True)

# non-blocking
gui = pyxalign.gui.launch_volume_histogram_viewer(volume)
```
