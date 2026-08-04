"""
GUI for inspecting a `PMASequence` recorded by
`LaminographyAlignmentTask.get_projection_matching_shift`.

Lets the user scroll through the captured `PMASnapshot` records,
visualize the initial and final shifts (sorted by angle, optionally
shifted by the previously applied shifts), and see which configuration
parameters varied across the sequence with the differing values
emphasized.
"""

import dataclasses
import sys
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

if TYPE_CHECKING:
    from pyxalign.data_structures.task import LaminographyAlignmentTask

# Pinned colours for the shift traces — chosen to match the previous
# matplotlib output ("gray" and "tab:blue") so the look is unchanged.
_INITIAL_SHIFT_PEN = pg.mkPen(color=(128, 128, 128), width=2)
_FINAL_SHIFT_PEN = pg.mkPen(color=(31, 119, 180), width=2)
_ANGLES_PEN = pg.mkPen(color=(31, 119, 180), width=2)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyxalign.alignment.pma_tracking import (
    PMASequence,
    PMASequencePlotType,
    PMASequenceSortAxis,
    PMASnapshot,
    compute_chains,
)
from pyxalign.interactions.viewers.base import ArrayViewer

_ALL_SNAPSHOTS_KEY = -1

# Field names of `PMASnapshot` whose values are configuration-style
# (compared across snapshots to highlight what changed). Per-iteration
# arrays such as `initial_shift`, `final_shift`, `past_shift_sum`,
# `angles`, and `scan_numbers` are excluded on purpose.
_DATACLASS_OPTION_FIELDS = (
    "pma_options",
    "reconstruct",
    "volume_width",
    "experiment",
    "mask_from_positions",
    "masks_from_roi",
)
_SCALAR_OPTION_FIELDS = ("center_of_rotation", "mask_source")

# Inside pma_options.reconstruct only these subfields are interesting enough
# to surface in the PMA-options tree; the rest are hidden.
_PMA_RECONSTRUCT_ALLOWED = {"geometry"}

# Phase-projection option fields that logically belong with the PMA settings
# (shown alongside pma_options) instead of in the "Other Options" tab.
_PMA_TAB_EXTRA_FIELDS = ("volume_width",)

# Dotted paths surfaced in the "3D Volume Options" tab. Each entry is
# resolved against a PMASnapshot — leaf values are shown as a single row,
# dataclass values are shown as an expanded subtree.
_VOLUME_TAB_PATHS = (
    "experiment.sample_thickness",
    "volume_width",
    "reconstruct.geometry",
    "center_of_rotation",
)

# Dotted paths surfaced in the "Mask Options" tab.
_MASK_TAB_PATHS = (
    "mask_source",
    "mask_from_positions",
    "masks_from_roi",
)

_HIGHLIGHT_COLOR = QColor("#b00020")


def _walk_dotted_path(obj: Any, dotted_path: str) -> Any:
    """Resolve a dotted attribute path against `obj`, e.g. 'experiment.sample_thickness'."""
    cur = obj
    for part in dotted_path.split("."):
        cur = getattr(cur, part)
    return cur


def _flatten_dataclass(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Return a flat {dotted_path: leaf_value} mapping for a (possibly nested) dataclass."""
    out: dict[str, Any] = {}
    if not dataclasses.is_dataclass(obj):
        out[prefix] = obj
        return out
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        path = f"{prefix}.{f.name}" if prefix else f.name
        if dataclasses.is_dataclass(value):
            out.update(_flatten_dataclass(value, path))
        else:
            out[path] = value
    return out


def _flatten_snapshot(snapshot: PMASnapshot) -> dict[str, Any]:
    """Flatten the configuration-style fields of a snapshot into dotted paths."""
    flat: dict[str, Any] = {}
    for name in _DATACLASS_OPTION_FIELDS:
        flat.update(_flatten_dataclass(getattr(snapshot, name), prefix=name))
    for name in _SCALAR_OPTION_FIELDS:
        flat[name] = getattr(snapshot, name)
    return flat


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.array_equal(np.asarray(a), np.asarray(b))
        except Exception:
            return False
    return a == b


def compute_varying_paths(snapshots: list[PMASnapshot]) -> set[str]:
    """Dotted paths whose value is not identical across every snapshot."""
    if len(snapshots) < 2:
        return set()
    flats = [_flatten_snapshot(s) for s in snapshots]
    keys = set().union(*(set(f.keys()) for f in flats))
    varying: set[str] = set()
    for key in keys:
        first = flats[0].get(key, _MISSING)
        for f in flats[1:]:
            if not _values_equal(first, f.get(key, _MISSING)):
                varying.add(key)
                break
    return varying


_MISSING = object()


def _format_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        if value.size <= 6:
            return np.array2string(value, precision=4, separator=", ")
        return f"<ndarray shape={value.shape} dtype={value.dtype}>"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


_HEADING_COLORS = {1: QColor("#060C78"), 2: QColor("#143dad")}


def _style_headings(browser: "QTextBrowser") -> None:
    """Apply colors to h1/h2 blocks after markdown is loaded."""
    block = browser.document().begin()
    while block.isValid():
        level = block.blockFormat().headingLevel()
        if level in _HEADING_COLORS:
            fmt = QTextCharFormat()
            fmt.setForeground(_HEADING_COLORS[level])
            cursor = QTextCursor(block)
            cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            cursor.mergeCharFormat(fmt)
        block = block.next()


_VIEWER_HELP_TEXT = """\
# PMA Sequence Viewer

Inspect every `get_projection_matching_shift` call captured in
`task.pma_sequence`.

## Snapshots table

- One row per PMA call. Click a row to load it into the right-hand display.
- **# Changed**: how many configuration parameters in this snapshot differ
  from the first snapshot in the current chain.
- **Initial shift from**: the prior snapshot whose `final_shift` was used to
  seed this run, or `N/A` if no parent was recorded.

## Chain dropdown

A chain is a linear path through `parent_index` links. 
- **All snapshots** — the table shows entries for every time projection-matching
    alignment was run. 
- **Chain to most recent #N** — the table shows entries for the alignment sequence 
    up to the latest run.
- **Chain to #M** — the table shows entries for a previous chain.

The "Changed Parameters" highlights and the "# Changed" counts are
recomputed against whichever chain is selected.

## Changed Parameters

Lists every configuration parameter whose value isn't identical across the
visible snapshots, plus any scans that were dropped before the selected
snapshot was recorded.

## Options for Selected PMA Snapshot

Three tabs show the inputs that shaped the selected run:
- **PMA Options** — the `ProjectionMatchingOptions` passed to PMA.
- **3D Volume Options** — `experiment.sample_thickness`, `volume_width`,
  `reconstruct.geometry`, `center_of_rotation`.
- **Mask Options** — `mask_source`, `mask_from_positions`, `masks_from_roi`.

Parameters that vary across the visible chain are bolded in red.

## Display

Pick what to plot for the selected snapshot:
- **initial / final shifts** - 
- **angles** — angle vs. scan number.
- **volume** — array viewer of the recorded post-PMA volume (only available
  if `pma_options.pma_sequence.record_volume` was on when PMA ran).

## Plot controls (shifts plot)

- **Sort x axis by** — angle or scan number.
- **Include past shifts from projections** — adds any shifts that have previously 
    been applied to the projections, so that you are seeing the absolute alignment 
    instead of the delta from this run's starting point.
- **Lock axis ranges across snapshots** — freezes the current x and y
  ranges so switching snapshots doesn't autoscale.
"""


_GENERAL_HELP_TEXT = """\
# General Help

## Tips

- Drag the splitter handles between **Snapshots**, **Changed Parameters**,
  and **Options for Selected PMA Snapshot** to resize each section.
- Hover over field names in the option trees to see their dotted path.
- The viewer is read-only: closing it never modifies the underlying
  `PMASequence` or task.
"""


class _HelpDialog(QDialog):
    """Sidebar-based help/tips dialog with viewer and general content."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Help / Tips")
        self.resize(640, 480)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        pages = [
            ("Viewer", _VIEWER_HELP_TEXT),
            ("General Help", _GENERAL_HELP_TEXT),
        ]

        outer_layout = QHBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.setLayout(outer_layout)

        # Left sidebar
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(130)
        self.sidebar.setStyleSheet(
            "QListWidget { background-color: #2e2e2e; border: none; }"
            "QListWidget::item { color: #cccccc; padding: 10px 12px; border: none; }"
            "QListWidget::item:selected { background-color: #444; color: white; }"
            "QListWidget::item:hover:!selected { background-color: #3a3a3a; }"
        )
        for name, _ in pages:
            self.sidebar.addItem(name)
        outer_layout.addWidget(self.sidebar)

        # Right side: content area + font size controls
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        outer_layout.addLayout(right_layout)

        self._font_size = 10
        self._browsers: list[QTextBrowser] = []
        self._texts: list[str] = []
        self.stacked = QStackedWidget()
        for _, text in pages:
            browser = QTextBrowser()
            browser.setReadOnly(True)
            browser.setStyleSheet("QTextBrowser { padding: 12px; }")
            self.stacked.addWidget(browser)
            self._browsers.append(browser)
            self._texts.append(text)
        self._refresh_browsers()
        right_layout.addWidget(self.stacked)

        font_bar = QHBoxLayout()
        font_bar.setContentsMargins(8, 4, 8, 4)
        font_bar.addStretch()
        for label, slot in (("−", self._zoom_out), ("+", self._zoom_in)):
            btn = QPushButton(label)
            btn.setFixedWidth(36)
            btn.setStyleSheet("QPushButton { font-weight: bold; }")
            btn.clicked.connect(slot)
            font_bar.addWidget(btn)
        right_layout.addLayout(font_bar)

        self.sidebar.currentRowChanged.connect(self.stacked.setCurrentIndex)
        self.sidebar.setCurrentRow(0)

    def _refresh_browsers(self) -> None:
        font = QFont()
        font.setPointSize(self._font_size)
        for browser, text in zip(self._browsers, self._texts):
            browser.setFont(font)
            browser.setMarkdown(text)
            _style_headings(browser)

    def _zoom_in(self) -> None:
        self._font_size += 1
        self._refresh_browsers()

    def _zoom_out(self) -> None:
        self._font_size = max(6, self._font_size - 1)
        self._refresh_browsers()


class PMASequenceViewer(QWidget):
    """Scrollable inspector for a `PMASequence`."""

    def __init__(
        self,
        sequence: PMASequence,
        parent: Optional[QWidget] = None,
        task: Optional["LaminographyAlignmentTask"] = None,
        projection_viewer: Optional[QWidget] = None,
        on_shift_staged: Optional[Callable[[], None]] = None,
    ):
        super().__init__(parent=parent)
        self.setWindowTitle("PMA Sequence Viewer")
        self.sequence = sequence
        self.task = task
        self.projection_viewer = projection_viewer
        # Callback invoked after a snapshot's shift is successfully staged
        # and applied — used by the PMA runner to clear stale alignment
        # results.
        self.on_shift_staged = on_shift_staged
        self.varying_paths: set[str] = set()
        self.show_with_past_shifts = False
        self.plot_type = PMASequencePlotType.INITIAL_FINAL_SHIFTS
        self.sort_axis = PMASequenceSortAxis.BY_ANGLE
        self._current_row: Optional[int] = None
        # Indices of snapshots currently shown in the table (after the
        # chain filter is applied). Empty means "show everything".
        self._visible_indices: list[int] = []
        self._chain_filter_terminal: int = _ALL_SNAPSHOTS_KEY
        # Snapshot whose volume is currently displayed.  Gives access to the
        # prior scale and crop offsets needed to map the view center when
        # switching to a new snapshot with different scale/crop settings.
        self._last_volume_snapshot: Optional[PMASnapshot] = None

        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        # Left: chain-filter dropdown + table + changed-params + tabbed details
        self.chain_combo = QComboBox()
        self.chain_combo.currentIndexChanged.connect(self._on_chain_filter_changed)

        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(
            ["Index", "# Changed", "Initial shift from", "Timestamp"]
        )
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.currentCellChanged.connect(self._on_row_changed)

        self.changed_label = QLabel("<i>Select a snapshot to see what changed.</i>")
        self.changed_label.setWordWrap(True)
        self.changed_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.changed_label.setStyleSheet(
            "QLabel { background-color: #f7f7f7; padding: 8px; border: 1px solid #ccc; }"
        )
        self.changed_scroll = QScrollArea()
        self.changed_scroll.setWidget(self.changed_label)
        self.changed_scroll.setWidgetResizable(True)
        self.changed_scroll.setMinimumHeight(60)

        self.pma_options_tree = self._make_options_tree()
        self.volume_options_tree = self._make_options_tree()
        self.mask_options_tree = self._make_options_tree()

        self.snapshot_info = QWidget()
        self.snapshot_info_layout = QFormLayout(self.snapshot_info)

        info_tabs = QTabWidget()
        info_tabs.addTab(self.pma_options_tree, "PMA Options")
        info_tabs.addTab(self.volume_options_tree, "3D Volume Options")
        info_tabs.addTab(self.mask_options_tree, "Mask Options")
        info_tabs.addTab(self.snapshot_info, "Snapshot Info")

        # Build three vertically-resizable sections via a QSplitter so the
        # user can drag handles to grow/shrink each region.
        snapshots_section = QWidget()
        snapshots_layout = QVBoxLayout(snapshots_section)
        snapshots_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Snapshots")
        title.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #003366; }")
        snapshots_layout.addWidget(title)
        chain_row = QHBoxLayout()
        chain_row.addWidget(QLabel("Chain:"))
        chain_row.addWidget(self.chain_combo, stretch=1)
        snapshots_layout.addLayout(chain_row)
        snapshots_layout.addWidget(self.results_table)

        self.stage_shift_button = QPushButton("Stage && Apply Selected Snapshot Shift")
        self.stage_shift_button.setStyleSheet(
            "QPushButton { background-color: #b3d9ff; font-weight: bold; padding: 6px; }"
            "QPushButton:disabled { background-color: #e0e0e0; color: #888; }"
        )
        self.stage_shift_button.clicked.connect(self._on_stage_shift_clicked)
        if self.task is None:
            self.stage_shift_button.setEnabled(False)
            self.stage_shift_button.setToolTip(
                "No task is wired up to this viewer — staging is unavailable."
            )
        snapshots_layout.addWidget(self.stage_shift_button)

        changed_section = QWidget()
        changed_layout = QVBoxLayout(changed_section)
        changed_layout.setContentsMargins(0, 0, 0, 0)
        changed_title = QLabel("Changed Parameters")
        changed_title.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #003366; }")
        changed_layout.addWidget(changed_title)
        changed_layout.addWidget(self.changed_scroll)

        options_section = QWidget()
        options_layout = QVBoxLayout(options_section)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_title = QLabel("Options for Selected PMA Snapshot")
        options_title.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #003366; }")
        options_layout.addWidget(options_title)
        options_layout.addWidget(info_tabs)

        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.addWidget(snapshots_section)
        left_splitter.addWidget(changed_section)
        left_splitter.addWidget(options_section)
        # Lower the minimum size hints so the splitter actually honours
        # the requested initial proportions instead of being dominated by
        # the tab widget's natural minimum height.
        info_tabs.setMinimumHeight(120)
        self.changed_scroll.setMinimumHeight(120)

        # Initial proportions (snapshots : changed : options) — bias toward
        # a larger Changed Parameters box and a smaller Options box.
        left_splitter.setSizes([240, 320, 200])
        left_splitter.setStretchFactor(0, 2)
        left_splitter.setStretchFactor(1, 3)
        left_splitter.setStretchFactor(2, 2)

        left_column = QVBoxLayout()
        left_column.addWidget(left_splitter)

        # Right: stacked display area (shifts / angles / volume) + plot controls.
        # Shifts: a single GraphicsLayoutWidget with two stacked plots.
        self.shifts_widget = pg.GraphicsLayoutWidget()
        self.shifts_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_horizontal = self.shifts_widget.addPlot(row=0, col=0, title="horizontal shifts")
        self.plot_vertical = self.shifts_widget.addPlot(row=1, col=0, title="vertical shifts")
        for plot in (self.plot_horizontal, self.plot_vertical):
            plot.setLabel("left", "shift (px)")
            plot.showGrid(x=True, y=True, alpha=0.3)
        # Link x-axes so panning/zooming the two shift plots stays in sync.
        self.plot_vertical.setXLink(self.plot_horizontal)

        # Angles: a single PlotWidget.
        self.angles_widget = pg.PlotWidget(title="angles vs scan number")
        self.angles_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.angles_widget.setLabel("bottom", "scan number")
        self.angles_widget.setLabel("left", "angle (deg)")
        self.angles_widget.showGrid(x=True, y=True, alpha=0.3)

        self.array_viewer = ArrayViewer(hide_axis_controls=False)

        self.display_stack = QStackedWidget()
        self.display_stack.addWidget(self.shifts_widget)   # 0: shifts
        self.display_stack.addWidget(self.angles_widget)   # 1: angles
        self.display_stack.addWidget(self.array_viewer)    # 2: volume

        # Plot type dropdown
        self.plot_type_combo = QComboBox()
        self._plot_type_order = [
            PMASequencePlotType.INITIAL_FINAL_SHIFTS,
            PMASequencePlotType.ANGLES,
            PMASequencePlotType.VOLUME,
        ]
        plot_type_labels = {
            PMASequencePlotType.INITIAL_FINAL_SHIFTS: "initial / final shifts",
            PMASequencePlotType.ANGLES: "angles",
            PMASequencePlotType.VOLUME: "volume",
        }
        for pt in self._plot_type_order:
            self.plot_type_combo.addItem(plot_type_labels[pt], pt)
        self.plot_type_combo.currentIndexChanged.connect(self._on_plot_type_changed)

        # Sort-axis radios (apply to shift/angle plots only)
        self.sort_group = QGroupBox("Sort x axis by")
        sort_layout = QHBoxLayout(self.sort_group)
        self.sort_button_group = QButtonGroup(self)
        self.sort_by_angle_button = QRadioButton("angle")
        self.sort_by_angle_button.setChecked(True)
        self.sort_by_scan_button = QRadioButton("scan number")
        for btn in (self.sort_by_angle_button, self.sort_by_scan_button):
            btn.setStyleSheet("font-size: 11pt;")
            btn.toggled.connect(self._on_sort_axis_changed)
            sort_layout.addWidget(btn)
            self.sort_button_group.addButton(btn)

        # Past-shift inclusion radios (only relevant for the shifts plot)
        self.view_group = QGroupBox("Shift offset")
        view_layout = QVBoxLayout(self.view_group)
        self.view_button_group = QButtonGroup(self)
        self.default_view_button = QRadioButton("Relative to initial (default)")
        self.default_view_button.setChecked(True)
        self.with_past_button = QRadioButton("Include past shifts from projections")
        for btn in (self.default_view_button, self.with_past_button):
            btn.setStyleSheet("font-size: 11pt;")
            btn.toggled.connect(self._on_view_mode_changed)
            view_layout.addWidget(btn)
            self.view_button_group.addButton(btn)
        # Lock both axes' ranges across snapshot changes (shifts plot only).
        self.lock_shift_axes_check = QCheckBox("Lock axis ranges across snapshots")
        self.lock_shift_axes_check.setStyleSheet("font-size: 11pt;")
        self.lock_shift_axes_check.toggled.connect(self._on_lock_axes_toggled)
        view_layout.addWidget(self.lock_shift_axes_check)

        # Plot controls bar
        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Display:"))
        controls_row.addWidget(self.plot_type_combo)
        controls_row.addStretch()

        right_column = QVBoxLayout()
        right_column.addLayout(controls_row)
        right_column.addWidget(self.display_stack, stretch=1)
        right_column.addWidget(self.sort_group)
        right_column.addWidget(self.view_group)

        # Top bar with action buttons anchored to the right.
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        self.delete_volume_button = QPushButton("Delete Volume Array")
        self.delete_volume_button.setStyleSheet(
            "QPushButton { background-color: #dc3545; color: white; font-weight: bold; }"
        )
        self.delete_volume_button.setToolTip(
            "Delete the volume array from the currently selected snapshot."
        )
        self.delete_volume_button.clicked.connect(self._on_delete_volume)
        top_bar.addWidget(self.delete_volume_button)
        self.help_button = QPushButton("Help / Tips")
        self.help_button.setStyleSheet(
            "QPushButton { background-color: #868e96; color: white; font-weight: bold; }"
        )
        self.help_button.setToolTip("Open help and tips for this window.")
        self.help_button.clicked.connect(self._on_help)
        top_bar.addWidget(self.help_button)

        body_layout = QHBoxLayout()
        body_layout.addLayout(left_column, stretch=1)
        body_layout.addLayout(right_column, stretch=3)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_bar)
        main_layout.addLayout(body_layout, stretch=1)

        self._sync_control_visibility()

    def _on_delete_volume(self) -> None:
        """Delete the volume array from the currently selected snapshot."""
        snapshot = self._current_snapshot()
        if snapshot is None or snapshot.volume is None:
            QMessageBox.information(
                self,
                "No Volume",
                "The current snapshot has no volume array to delete.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete Volume Array",
            "Are you sure you want to delete the volume array from this snapshot? "
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        snap_idx = self._row_to_snapshot_index(self._current_row)
        if snap_idx is None:
            return
        self.sequence.snapshots[snap_idx].volume = None
        self._update_volume_view(self.sequence.snapshots[snap_idx])

    def _on_help(self) -> None:
        """Open the Help/Tips dialog (non-modal, single instance)."""
        if not hasattr(self, "_help_dialog") or not self._help_dialog.isVisible():
            self._help_dialog = _HelpDialog(parent=self)
            self._help_dialog.show()
        else:
            self._help_dialog.raise_()
            self._help_dialog.activateWindow()

    # ---- public API -----------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the chain dropdown, table, and diffs after a sequence change."""
        self._rebuild_chain_combo()
        self._recompute_visible_indices()
        self._populate_table()
        if self._visible_indices:
            target = self._current_row if self._current_row is not None else 0
            target = min(target, len(self._visible_indices) - 1)
            self.results_table.selectRow(target)
        else:
            self._clear_displays()

    # ---- chain filter ---------------------------------------------------

    def _rebuild_chain_combo(self) -> None:
        chains = compute_chains(self.sequence.snapshots)
        # Sort terminals so the most recent (highest index) is first
        sorted_terminals = sorted(chains.keys(), reverse=True)
        most_recent = sorted_terminals[0] if sorted_terminals else None

        # Preserve the current selection if it still exists
        prior = self._chain_filter_terminal

        self.chain_combo.blockSignals(True)
        self.chain_combo.clear()
        self.chain_combo.addItem(
            f"All snapshots ({len(self.sequence)})", _ALL_SNAPSHOTS_KEY
        )
        for terminal in sorted_terminals:
            chain = chains[terminal]
            if terminal == most_recent:
                label = f"Chain to most recent #{terminal} ({len(chain)} snap)"
            elif len(chain) < 2:
                # Skip isolated snapshots from the "other sequences" listing.
                continue
            else:
                label = f"Chain to #{terminal} ({len(chain)} snap)"
            self.chain_combo.addItem(label, terminal)

        # Restore prior selection if it still exists, else default to "All".
        idx = self.chain_combo.findData(prior)
        self.chain_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._chain_filter_terminal = self.chain_combo.currentData()
        self.chain_combo.blockSignals(False)

    def _recompute_visible_indices(self) -> None:
        terminal = self._chain_filter_terminal
        if terminal == _ALL_SNAPSHOTS_KEY or terminal is None:
            self._visible_indices = list(range(len(self.sequence)))
        else:
            chains = compute_chains(self.sequence.snapshots)
            self._visible_indices = chains.get(terminal, list(range(len(self.sequence))))
        # Diffs are scoped to the visible subset so the highlights describe
        # what changed within the chain the user is currently inspecting.
        visible_snaps = [self.sequence.snapshots[i] for i in self._visible_indices]
        self.varying_paths = compute_varying_paths(visible_snaps)

    def _on_chain_filter_changed(self, _index: int) -> None:
        self._chain_filter_terminal = self.chain_combo.currentData()
        self._current_row = None
        self._recompute_visible_indices()
        self._populate_table()
        if self._visible_indices:
            self.results_table.selectRow(0)
        else:
            self._clear_displays()

    # ---- table ----------------------------------------------------------

    def _populate_table(self) -> None:
        self.results_table.setRowCount(0)
        if not self._visible_indices:
            return
        visible_snaps = [self.sequence.snapshots[i] for i in self._visible_indices]
        flats = [_flatten_snapshot(s) for s in visible_snaps]
        baseline = flats[0] if flats else {}
        for row, idx in enumerate(self._visible_indices):
            snap = self.sequence.snapshots[idx]
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(str(idx)))
            n_changed = sum(
                1
                for path in self.varying_paths
                if not _values_equal(flats[row].get(path, _MISSING), baseline.get(path, _MISSING))
            )
            self.results_table.setItem(row, 1, QTableWidgetItem(str(n_changed)))
            parent = getattr(snap, "parent_index", None)
            parent_text = "N/A" if parent is None else f"snapshot {parent}"
            self.results_table.setItem(row, 2, QTableWidgetItem(parent_text))
            self.results_table.setItem(row, 3, QTableWidgetItem(snap.timestamp or ""))

    def _row_to_snapshot_index(self, row: int) -> Optional[int]:
        if row < 0 or row >= len(self._visible_indices):
            return None
        return self._visible_indices[row]

    def _on_row_changed(self, row: int, _column: int) -> None:
        snap_idx = self._row_to_snapshot_index(row)
        if snap_idx is None:
            return
        self._current_row = row
        snapshot = self.sequence[snap_idx]
        self._update_display(snapshot)
        self._update_changed_label(snapshot)
        self._update_options_tree(snapshot)
        self._update_snapshot_info(snapshot)
        self._update_stage_button_state(snapshot)

    # ---- stage shift ----------------------------------------------------

    def _update_stage_button_state(self, snapshot: Optional[PMASnapshot]) -> None:
        """Disable the stage button if the snapshot has no usable final shift."""
        if self.task is None:
            # The no-task disabled state was set in `_build_ui`; nothing to do.
            return
        has_final = snapshot is not None and snapshot.final_shift is not None
        self.stage_shift_button.setEnabled(has_final)
        if has_final:
            self.stage_shift_button.setToolTip(
                "Stage and apply this snapshot's shift to the projections. "
                "The shift is automatically re-expressed relative to whatever "
                "has been applied since the snapshot was recorded."
            )
        else:
            self.stage_shift_button.setToolTip(
                "This snapshot has no final_shift (PMA did not complete)."
            )

    def _on_stage_shift_clicked(self) -> None:
        from pyxalign.api import enums

        if self.task is None or self.task.phase_projections is None:
            return
        snapshot = self._current_snapshot()
        if snapshot is None or snapshot.final_shift is None:
            return

        # Resolve the snapshot's absolute alignment into a delta relative
        # to the projections' current state. Catch the
        # "scan no longer in projections" / "scan number mismatch" cases
        # the helper raises so the user gets a clear message instead of
        # an unhandled exception.
        try:
            shift_to_stage = snapshot.compute_shift_relative_to(
                self.task.phase_projections
            )
        except ValueError as ex:
            QMessageBox.warning(self, "Cannot stage shift", str(ex))
            return

        reply = QMessageBox.question(
            self,
            "Confirm",
            "Projections will be shifted using this snapshot's final shift "
            "(accounting for any shifts already applied). Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.task.phase_projections.shift_manager.stage_shift(
            shift=shift_to_stage,
            function_type=enums.ShiftType.CIRC,
            alignment_options=self.task.options.projection_matching,
            eliminate_wrapping=True,
        )
        print("Snapshot shift staged from PMA Sequence Viewer")

        if self.projection_viewer is not None:
            self.projection_viewer.refresh_applied_shifts_tab()
            self.projection_viewer.apply_staged_shift()

        if self.on_shift_staged is not None:
            self.on_shift_staged()

    # ---- view mode ------------------------------------------------------

    def _current_snapshot(self) -> Optional[PMASnapshot]:
        snap_idx = self._row_to_snapshot_index(self._current_row) if self._current_row is not None else None
        return self.sequence[snap_idx] if snap_idx is not None else None

    def _on_view_mode_changed(self) -> None:
        self.show_with_past_shifts = self.with_past_button.isChecked()
        snapshot = self._current_snapshot()
        if snapshot is not None:
            self._update_display(snapshot)

    def _on_lock_axes_toggled(self, _checked: bool) -> None:
        # When the lock is released let the next refresh autorange; when
        # it's engaged, freeze the current x and y ranges so subsequent
        # snapshot selections don't rescale either axis.
        snapshot = self._current_snapshot()
        if not self.lock_shift_axes_check.isChecked():
            for plot in (self.plot_horizontal, self.plot_vertical):
                plot.enableAutoRange(axis="xy", enable=True)
        if snapshot is not None and self.plot_type == PMASequencePlotType.INITIAL_FINAL_SHIFTS:
            self._update_display(snapshot)

    def _on_plot_type_changed(self, _index: int) -> None:
        self.plot_type = self.plot_type_combo.currentData()
        self._sync_control_visibility()
        snapshot = self._current_snapshot()
        if snapshot is not None:
            self._update_display(snapshot)

    def _on_sort_axis_changed(self) -> None:
        self.sort_axis = (
            PMASequenceSortAxis.BY_ANGLE
            if self.sort_by_angle_button.isChecked()
            else PMASequenceSortAxis.BY_SCAN_NUMBER
        )
        snapshot = self._current_snapshot()
        if snapshot is not None:
            self._update_display(snapshot)

    def _sync_control_visibility(self) -> None:
        is_shifts = self.plot_type == PMASequencePlotType.INITIAL_FINAL_SHIFTS
        is_volume = self.plot_type == PMASequencePlotType.VOLUME
        # Sort axis is only meaningful for the shifts plot. The angles plot
        # is always shown vs. scan number; the volume display has no x axis.
        self.sort_group.setVisible(is_shifts)
        self.view_group.setVisible(is_shifts)
        self.delete_volume_button.setVisible(is_volume)

    # ---- display dispatch -----------------------------------------------

    def _get_sort_idx(self, snapshot: PMASnapshot) -> np.ndarray:
        if self.sort_axis == PMASequenceSortAxis.BY_ANGLE:
            return np.argsort(np.asarray(snapshot.angles))
        return np.argsort(np.asarray(snapshot.scan_numbers))

    def _x_axis_values_and_label(
        self, snapshot: PMASnapshot, sort_idx: np.ndarray
    ) -> tuple[np.ndarray, str]:
        if self.sort_axis == PMASequenceSortAxis.BY_ANGLE:
            return np.asarray(snapshot.angles)[sort_idx], "angle (deg)"
        return np.asarray(snapshot.scan_numbers)[sort_idx], "scan number"

    def _update_display(self, snapshot: PMASnapshot) -> None:
        if self.plot_type == PMASequencePlotType.INITIAL_FINAL_SHIFTS:
            self.display_stack.setCurrentWidget(self.shifts_widget)
            self._update_shifts_plot(snapshot)
        elif self.plot_type == PMASequencePlotType.ANGLES:
            self.display_stack.setCurrentWidget(self.angles_widget)
            self._update_angles_plot(snapshot)
        elif self.plot_type == PMASequencePlotType.VOLUME:
            self.display_stack.setCurrentWidget(self.array_viewer)
            self._update_volume_view(snapshot)

    def _update_shifts_plot(self, snapshot: PMASnapshot) -> None:
        sort_idx = self._get_sort_idx(snapshot)
        x_values, x_label = self._x_axis_values_and_label(snapshot, sort_idx)

        initial = snapshot.initial_shift
        final = snapshot.final_shift
        if self.show_with_past_shifts:
            offset = snapshot.past_shift_sum
            if initial is not None:
                initial = np.asarray(initial) + offset
            if final is not None:
                final = np.asarray(final) + offset

        lock_axes = self.lock_shift_axes_check.isChecked()
        # Capture the current x and y ranges before we clear the plots; if
        # the lock is engaged we restore them afterward instead of
        # autoranging.
        locked_ranges = []
        if lock_axes:
            for plot in (self.plot_horizontal, self.plot_vertical):
                xrange, yrange = plot.getViewBox().viewRange()
                locked_ranges.append((tuple(xrange), tuple(yrange)))

        for i, plot in enumerate([self.plot_horizontal, self.plot_vertical]):
            plot.clear()
            # pyqtgraph re-creates the legend per addLegend() call; clear()
            # already wipes the old one.
            plot.addLegend(offset=(10, 10))
            plot.setLabel("bottom", x_label)
            if initial is not None:
                plot.plot(
                    np.asarray(x_values),
                    np.asarray(initial)[sort_idx, i],
                    pen=_INITIAL_SHIFT_PEN,
                    name="initial",
                )
            if final is not None:
                plot.plot(
                    np.asarray(x_values),
                    np.asarray(final)[sort_idx, i],
                    pen=_FINAL_SHIFT_PEN,
                    name="final",
                )
            if lock_axes:
                (xmin, xmax), (ymin, ymax) = locked_ranges[i]
                plot.enableAutoRange(axis="xy", enable=False)
                plot.setXRange(xmin, xmax, padding=0)
                plot.setYRange(ymin, ymax, padding=0)
            else:
                plot.enableAutoRange(axis="xy", enable=True)

    def _update_angles_plot(self, snapshot: PMASnapshot) -> None:
        sort_idx = np.argsort(np.asarray(snapshot.scan_numbers))
        x_values = np.asarray(snapshot.scan_numbers)[sort_idx]
        y_values = np.asarray(snapshot.angles)[sort_idx]

        self.angles_widget.clear()
        self.angles_widget.plot(
            x_values,
            y_values,
            pen=_ANGLES_PEN,
            symbol="o",
            symbolSize=5,
            symbolBrush=(31, 119, 180),
            symbolPen=None,
        )
        self.angles_widget.enableAutoRange(axis="xy", enable=True)

    def _update_volume_view(self, snapshot: PMASnapshot) -> None:
        volume = snapshot.volume
        # Preserve the fractional slice position so switching snapshots keeps
        # the viewer oriented at the same relative depth in the new volume.
        current_max = self.array_viewer.slider.maximum()
        slice_fraction = (
            self.array_viewer.slider.value() / current_max if current_max > 0 else 0.0
        )
        # Capture view state and prior volume metadata before reinitializing.
        view_box = self.array_viewer.plot_item.getViewBox()
        x_range, y_range = view_box.viewRange()
        prior_snapshot = self._last_volume_snapshot
        prior_vol_shape = (
            self.array_viewer.array3d.shape
            if self.array_viewer.array3d is not None
            else None
        )

        if volume is None or np.asarray(volume).size == 0:
            self._last_volume_snapshot = None
            empty = np.zeros((1, 1, 1), dtype=np.float32)
            self.array_viewer.reinitialize_all(array3d=empty)
            self.array_viewer.setToolTip(
                "No volume was recorded for this snapshot. "
                "Set pma_options.pma_sequence.record_volume = True before running PMA."
            )
            return

        self.array_viewer.setToolTip("")
        self.array_viewer.reinitialize_all(array3d=np.asarray(volume))
        self._last_volume_snapshot = snapshot

        new_max = self.array_viewer.slider.maximum()
        if new_max > 0:
            self.array_viewer.slider.setValue(round(slice_fraction * new_max))

        if prior_snapshot is None or prior_vol_shape is None:
            # First real volume — let auto-range position it.
            return

        # Map the current view center from the old volume's pixel space to the
        # new volume's pixel space, preserving apparent physical position.
        #
        # For a volume at scale s with crop (horizontal_offset=ox, vertical_offset=oy),
        # a viewer pixel p maps to a position relative to the full reconstruction
        # center of:  (p - image_half + crop_offset) * s
        # Inverting gives the mapping A → B:
        #   rel = (p_A - W_A/2 + ox_A) * s_A / s_B
        #   p_B = rel + W_B/2 - ox_B
        # Half-widths scale by s_A / s_B to preserve physical extent (zoom level).
        # When crop is disabled, offset is 0; the formula degenerates correctly.
        slider_axis = self.array_viewer.options.slider_axis

        old_ns = [d for i, d in enumerate(prior_vol_shape) if i != slider_axis]
        new_ns = [d for i, d in enumerate(self.array_viewer.array3d.shape) if i != slider_axis]
        # display_frame transposes the slice: non_slider[0] → viewer-y, non_slider[1] → viewer-x
        old_W, old_H = float(old_ns[1]), float(old_ns[0])
        new_W, new_H = float(new_ns[1]), float(new_ns[0])

        s_A = float(prior_snapshot.pma_options.downsample.scale)
        s_B = float(snapshot.pma_options.downsample.scale)

        old_crop = prior_snapshot.pma_options.pma_sequence.volume_crop
        ox_A = float(old_crop.horizontal_offset) if old_crop.enabled else 0.0
        oy_A = float(old_crop.vertical_offset) if old_crop.enabled else 0.0

        new_crop = snapshot.pma_options.pma_sequence.volume_crop
        ox_B = float(new_crop.horizontal_offset) if new_crop.enabled else 0.0
        oy_B = float(new_crop.vertical_offset) if new_crop.enabled else 0.0

        x_center_A = (x_range[0] + x_range[1]) / 2
        y_center_A = (y_range[0] + y_range[1]) / 2
        x_half = (x_range[1] - x_range[0]) / 2 * s_A / s_B
        y_half = (y_range[1] - y_range[0]) / 2 * s_A / s_B

        rel_x = (x_center_A - old_W / 2 + ox_A) * s_A / s_B
        rel_y = (y_center_A - old_H / 2 + oy_A) * s_A / s_B
        x_center_B = rel_x + new_W / 2 - ox_B
        y_center_B = rel_y + new_H / 2 - oy_B

        view_box.setRange(
            xRange=(x_center_B - x_half, x_center_B + x_half),
            yRange=(y_center_B - y_half, y_center_B + y_half),
            padding=0,
        )

    # ---- changed-params label ------------------------------------------

    def _update_changed_label(self, snapshot: PMASnapshot) -> None:
        lines: list[str] = []

        if self.varying_paths:
            lines.append("<b>Differs across the sequence:</b><br>")
            flat = _flatten_snapshot(snapshot)
            for path in sorted(self.varying_paths):
                value = flat.get(path, _MISSING)
                shown = "&lt;missing&gt;" if value is _MISSING else _format_value(value)
                lines.append(f"&bull; <b>{path}</b>: {shown}<br>")

        removed_scans = np.asarray(getattr(snapshot, "removed_scan_numbers", []))
        removed_angles = np.asarray(getattr(snapshot, "removed_angles", []))
        if removed_scans.size > 0:
            if lines:
                lines.append("<br>")
            lines.append(
                f"<b>Removed scans ({removed_scans.size}):</b><br>"
            )
            for scan, angle in zip(removed_scans.tolist(), removed_angles.tolist()):
                angle_str = "?" if angle != angle else f"{angle:.4f}"  # NaN check
                lines.append(f"&nbsp;&nbsp;&bull; scan {int(scan)} (angle {angle_str})<br>")

        if not lines:
            self.changed_label.setText(
                "<i>All snapshots use identical configuration parameters and no scans have been removed.</i>"
            )
            return
        self.changed_label.setText("".join(lines))

    # ---- tree -----------------------------------------------------------

    def _make_options_tree(self) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Field", "Value"])
        tree.setAlternatingRowColors(True)
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        return tree

    def _populate_tree_from_root(
        self,
        tree: QTreeWidget,
        snapshot: PMASnapshot,
        dataclass_fields: tuple[str, ...],
        scalar_fields: tuple[str, ...],
    ) -> None:
        tree.clear()
        snap_idx = self._row_to_snapshot_index(self._current_row) if self._current_row is not None else None
        root = QTreeWidgetItem(tree, [f"snapshot[{snap_idx}]"])
        font = root.font(0)
        font.setBold(True)
        root.setFont(0, font)

        for name in dataclass_fields:
            value = getattr(snapshot, name)
            child = QTreeWidgetItem(root, [name])
            child_font = child.font(0)
            child_font.setBold(True)
            child.setFont(0, child_font)
            self._add_dataclass_to_tree(child, value, prefix=name)

        for name in scalar_fields:
            value = getattr(snapshot, name)
            leaf = QTreeWidgetItem(root, [name, _format_value(value)])
            leaf.setToolTip(0, name)
            self._maybe_highlight_leaf(leaf, name)

        tree.expandAll()

    def _update_options_tree(self, snapshot: PMASnapshot) -> None:
        # Tab 1: pma_options plus phase-projection option fields that
        # logically belong with the PMA settings (e.g. volume_width).
        pma_dataclass = ("pma_options",) + _PMA_TAB_EXTRA_FIELDS
        self._populate_tree_from_root(
            self.pma_options_tree,
            snapshot,
            dataclass_fields=pma_dataclass,
            scalar_fields=(),
        )
        # Tab 2: only the volume-shaping fields.
        self._populate_tree_from_paths(
            self.volume_options_tree, snapshot, _VOLUME_TAB_PATHS
        )
        # Tab 3: only the mask-related fields.
        self._populate_tree_from_paths(
            self.mask_options_tree, snapshot, _MASK_TAB_PATHS
        )

    def _populate_tree_from_paths(
        self,
        tree: QTreeWidget,
        snapshot: PMASnapshot,
        paths: tuple[str, ...],
    ) -> None:
        """Render only the explicitly-listed dotted paths from the snapshot."""
        tree.clear()
        snap_idx = self._row_to_snapshot_index(self._current_row) if self._current_row is not None else None
        root = QTreeWidgetItem(tree, [f"snapshot[{snap_idx}]"])
        font = root.font(0)
        font.setBold(True)
        root.setFont(0, font)

        for path in paths:
            try:
                value = _walk_dotted_path(snapshot, path)
            except AttributeError:
                continue
            if dataclasses.is_dataclass(value):
                child = QTreeWidgetItem(root, [path])
                child_font = child.font(0)
                child_font.setBold(True)
                child.setFont(0, child_font)
                child.setToolTip(0, path)
                self._add_dataclass_to_tree(child, value, prefix=path)
            else:
                leaf = QTreeWidgetItem(root, [path, _format_value(value)])
                leaf.setToolTip(0, path)
                self._maybe_highlight_leaf(leaf, path)

        tree.expandAll()

    def _add_dataclass_to_tree(
        self, parent_item: QTreeWidgetItem, data: Any, prefix: str
    ) -> None:
        if not dataclasses.is_dataclass(data):
            parent_item.setText(1, _format_value(data))
            self._maybe_highlight_leaf(parent_item, prefix)
            return

        non_dc: list[tuple[str, Any]] = []
        dc: list[tuple[str, Any]] = []
        for f in dataclasses.fields(data):
            # Inside pma_options.reconstruct only the whitelisted children
            # (e.g. geometry, volume_width) are surfaced.
            if (
                prefix == "pma_options.reconstruct"
                and f.name not in _PMA_RECONSTRUCT_ALLOWED
            ):
                continue
            v = getattr(data, f.name)
            (dc if dataclasses.is_dataclass(v) else non_dc).append((f.name, v))

        for name, value in non_dc:
            path = f"{prefix}.{name}"
            leaf = QTreeWidgetItem(parent_item, [name, _format_value(value)])
            leaf.setToolTip(0, path)
            self._maybe_highlight_leaf(leaf, path)

        for name, value in dc:
            path = f"{prefix}.{name}"
            child = QTreeWidgetItem(parent_item, [name])
            child_font = child.font(0)
            child_font.setBold(True)
            child.setFont(0, child_font)
            child.setToolTip(0, path)
            self._add_dataclass_to_tree(child, value, prefix=path)

    def _maybe_highlight_leaf(self, item: QTreeWidgetItem, path: str) -> None:
        if path not in self.varying_paths:
            return
        brush = QBrush(_HIGHLIGHT_COLOR)
        for col in (0, 1):
            item.setForeground(col, brush)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.setFont(1, font)

    # ---- snapshot info form --------------------------------------------

    def _update_snapshot_info(self, snapshot: PMASnapshot) -> None:
        # Clear the form
        while self.snapshot_info_layout.rowCount():
            self.snapshot_info_layout.removeRow(0)

        def add(label: str, value: str) -> None:
            v = QLabel(value)
            v.setWordWrap(True)
            v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.snapshot_info_layout.addRow(QLabel(label), v)

        add("Timestamp:", snapshot.timestamp or "<unknown>")
        add("# projections:", str(np.asarray(snapshot.angles).shape[0]))
        add("Mask source:", str(snapshot.mask_source) if snapshot.mask_source is not None else "None")
        cor = snapshot.center_of_rotation
        if cor is not None and np.asarray(cor).size >= 2:
            cor_arr = np.asarray(cor).ravel()
            add("Center of rotation (y, x):", f"({cor_arr[0]:.4f}, {cor_arr[1]:.4f})")
        add(
            "Initial shift:",
            "None" if snapshot.initial_shift is None else f"shape={np.asarray(snapshot.initial_shift).shape}",
        )
        add(
            "Final shift:",
            "None (PMA did not complete)"
            if snapshot.final_shift is None
            else f"shape={np.asarray(snapshot.final_shift).shape}",
        )
        add(
            "Past shift sum range:",
            (
                f"y in [{np.asarray(snapshot.past_shift_sum)[:, 0].min():.3f}, "
                f"{np.asarray(snapshot.past_shift_sum)[:, 0].max():.3f}], "
                f"x in [{np.asarray(snapshot.past_shift_sum)[:, 1].min():.3f}, "
                f"{np.asarray(snapshot.past_shift_sum)[:, 1].max():.3f}]"
            ),
        )
        removed = np.asarray(getattr(snapshot, "removed_scan_numbers", []))
        add("# removed scans:", str(removed.size))
        if snapshot.volume is None:
            add("Recorded volume:", "None")
        else:
            vol = np.asarray(snapshot.volume)
            add("Recorded volume:", f"shape={vol.shape} dtype={vol.dtype}")

    def _clear_displays(self) -> None:
        self._current_row = None
        for plot in (self.plot_horizontal, self.plot_vertical):
            plot.clear()
        self.angles_widget.clear()
        self.array_viewer.update_arrays(np.zeros((1, 1, 1), dtype=np.float32))
        self.changed_label.setText("<i>Sequence is empty.</i>")
        self.pma_options_tree.clear()
        self.volume_options_tree.clear()
        self.mask_options_tree.clear()
        while self.snapshot_info_layout.rowCount():
            self.snapshot_info_layout.removeRow(0)
        self._update_stage_button_state(None)


def show_pma_sequence_viewer(sequence: PMASequence) -> PMASequenceViewer:
    """Open the viewer in a standalone window. Runs the Qt event loop if needed."""
    app = QApplication.instance()
    own_app = app is None
    if own_app:
        app = QApplication(sys.argv)
    viewer = PMASequenceViewer(sequence)
    viewer.resize(1200, 700)
    viewer.show()
    if own_app:
        app.exec_()
    return viewer


def launch_pma_sequence_viewer(
    task: "LaminographyAlignmentTask",
    projection_viewer: Optional[QWidget] = None,
    wait_until_closed: bool = False,
) -> PMASequenceViewer:
    """Launch the PMA Sequence Viewer for a task.

    The viewer is opened with the task wired up so snapshot shifts can
    be staged and applied directly. Pass `projection_viewer` if you
    want the viewer to also drive the projection viewer's
    Applied Shifts tab when staging.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = PMASequenceViewer(
        task.pma_sequence,
        task=task,
        projection_viewer=projection_viewer,
    )
    viewer.resize(1200, 700)
    viewer.setAttribute(Qt.WA_DeleteOnClose)
    viewer.show()
    if wait_until_closed:
        app.exec_()
    return viewer
