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
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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

_HIGHLIGHT_COLOR = QColor("#b00020")


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


class PMASequenceViewer(QWidget):
    """Scrollable inspector for a `PMASequence`."""

    def __init__(
        self,
        sequence: PMASequence,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent=parent)
        self.setWindowTitle("PMA Sequence Viewer")
        self.sequence = sequence
        self.varying_paths: set[str] = set()
        self.show_with_past_shifts = False
        self.plot_type = PMASequencePlotType.INITIAL_FINAL_SHIFTS
        self.sort_axis = PMASequenceSortAxis.BY_ANGLE
        self._current_row: Optional[int] = None
        # Indices of snapshots currently shown in the table (after the
        # chain filter is applied). Empty means "show everything".
        self._visible_indices: list[int] = []
        self._chain_filter_terminal: int = _ALL_SNAPSHOTS_KEY

        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        # Left: chain-filter dropdown + table + changed-params + tabbed details
        self.chain_combo = QComboBox()
        self.chain_combo.currentIndexChanged.connect(self._on_chain_filter_changed)

        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(
            ["Index", "Timestamp", "# Changed", "Initial shift from"]
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
        changed_scroll = QScrollArea()
        changed_scroll.setWidget(self.changed_label)
        changed_scroll.setWidgetResizable(True)
        changed_scroll.setMinimumHeight(120)
        changed_scroll.setMaximumHeight(220)

        self.options_tree = QTreeWidget()
        self.options_tree.setColumnCount(2)
        self.options_tree.setHeaderLabels(["Field", "Value"])
        self.options_tree.setAlternatingRowColors(True)
        self.options_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)

        self.snapshot_info = QWidget()
        self.snapshot_info_layout = QFormLayout(self.snapshot_info)

        info_tabs = QTabWidget()
        info_tabs.addTab(self.options_tree, "All Settings")
        info_tabs.addTab(self.snapshot_info, "Snapshot Info")

        left_column = QVBoxLayout()
        title = QLabel("Snapshots")
        title.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; }")
        left_column.addWidget(title)
        chain_row = QHBoxLayout()
        chain_row.addWidget(QLabel("Chain:"))
        chain_row.addWidget(self.chain_combo, stretch=1)
        left_column.addLayout(chain_row)
        left_column.addWidget(self.results_table)
        changed_title = QLabel("Changed Parameters")
        changed_title.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; }")
        left_column.addWidget(changed_title)
        left_column.addWidget(changed_scroll)
        left_column.addWidget(info_tabs, stretch=1)

        # Right: stacked display area (canvas / volume viewer) + plot controls
        self.figure = Figure(figsize=(5, 4), layout="compressed")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax_horizontal = self.figure.add_subplot(211)
        self.ax_vertical = self.figure.add_subplot(212)

        self.angles_figure = Figure(figsize=(5, 4), layout="compressed")
        self.angles_canvas = FigureCanvas(self.angles_figure)
        self.angles_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax_angles = self.angles_figure.add_subplot(111)

        self.array_viewer = ArrayViewer(hide_axis_controls=False)

        self.display_stack = QStackedWidget()
        self.display_stack.addWidget(self.canvas)          # 0: shifts
        self.display_stack.addWidget(self.angles_canvas)   # 1: angles
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

        main_layout = QHBoxLayout(self)
        main_layout.addLayout(left_column, stretch=1)
        main_layout.addLayout(right_column, stretch=3)

        self._sync_control_visibility()

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
            self.results_table.setItem(row, 1, QTableWidgetItem(snap.timestamp or ""))
            n_changed = sum(
                1
                for path in self.varying_paths
                if not _values_equal(flats[row].get(path, _MISSING), baseline.get(path, _MISSING))
            )
            self.results_table.setItem(row, 2, QTableWidgetItem(str(n_changed)))
            parent = getattr(snap, "parent_index", None)
            parent_text = "N/A" if parent is None else f"snapshot {parent}"
            self.results_table.setItem(row, 3, QTableWidgetItem(parent_text))

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

    # ---- view mode ------------------------------------------------------

    def _current_snapshot(self) -> Optional[PMASnapshot]:
        snap_idx = self._row_to_snapshot_index(self._current_row) if self._current_row is not None else None
        return self.sequence[snap_idx] if snap_idx is not None else None

    def _on_view_mode_changed(self) -> None:
        self.show_with_past_shifts = self.with_past_button.isChecked()
        snapshot = self._current_snapshot()
        if snapshot is not None:
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
        # Sort axis is only meaningful for the shifts plot. The angles plot
        # is always shown vs. scan number; the volume display has no x axis.
        self.sort_group.setVisible(is_shifts)
        self.view_group.setVisible(is_shifts)

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
            self.display_stack.setCurrentWidget(self.canvas)
            self._update_shifts_plot(snapshot)
        elif self.plot_type == PMASequencePlotType.ANGLES:
            self.display_stack.setCurrentWidget(self.angles_canvas)
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

        for i, ax in enumerate([self.ax_horizontal, self.ax_vertical]):
            ax.clear()
            label = "horizontal" if i == 0 else "vertical"
            ax.set_title(f"{label} shifts")
            ax.set_xlabel(x_label)
            ax.set_ylabel("shift (px)")
            if initial is not None:
                ax.plot(x_values, np.asarray(initial)[sort_idx, i], label="initial")
            if final is not None:
                ax.plot(x_values, np.asarray(final)[sort_idx, i], label="final")
            ax.autoscale(enable=True, axis="x", tight=True)
            ax.grid(linestyle=":")
            if initial is not None or final is not None:
                ax.legend()
        self.canvas.draw()

    def _update_angles_plot(self, snapshot: PMASnapshot) -> None:
        sort_idx = np.argsort(np.asarray(snapshot.scan_numbers))
        x_values = np.asarray(snapshot.scan_numbers)[sort_idx]
        y_values = np.asarray(snapshot.angles)[sort_idx]

        self.ax_angles.clear()
        self.ax_angles.set_title("angles vs scan number")
        self.ax_angles.set_xlabel("scan number")
        self.ax_angles.set_ylabel("angle (deg)")
        self.ax_angles.plot(x_values, y_values, marker=".", linestyle="-")
        self.ax_angles.autoscale(enable=True, axis="x", tight=True)
        self.ax_angles.grid(linestyle=":")
        self.angles_canvas.draw()

    def _update_volume_view(self, snapshot: PMASnapshot) -> None:
        volume = snapshot.volume
        if volume is None or np.asarray(volume).size == 0:
            empty = np.zeros((1, 1, 1), dtype=np.float32)
            self.array_viewer.reinitialize_all(array3d=empty)
            self.array_viewer.setToolTip(
                "No volume was recorded for this snapshot. "
                "Set pma_options.pma_sequence.record_volume = True before running PMA."
            )
            return
        self.array_viewer.setToolTip("")
        self.array_viewer.reinitialize_all(array3d=np.asarray(volume))

    # ---- changed-params label ------------------------------------------

    def _update_changed_label(self, snapshot: PMASnapshot) -> None:
        lines: list[str] = []

        removed_scans = np.asarray(getattr(snapshot, "removed_scan_numbers", []))
        removed_angles = np.asarray(getattr(snapshot, "removed_angles", []))
        if removed_scans.size > 0:
            lines.append(
                f"<b>Removed scans ({removed_scans.size}):</b><br>"
            )
            for scan, angle in zip(removed_scans.tolist(), removed_angles.tolist()):
                angle_str = "?" if angle != angle else f"{angle:.4f}"  # NaN check
                lines.append(f"&nbsp;&nbsp;&bull; scan {int(scan)} (angle {angle_str})<br>")

        if self.varying_paths:
            if lines:
                lines.append("<br>")
            lines.append("<b>Differs across the sequence:</b><br>")
            flat = _flatten_snapshot(snapshot)
            for path in sorted(self.varying_paths):
                value = flat.get(path, _MISSING)
                shown = "&lt;missing&gt;" if value is _MISSING else _format_value(value)
                lines.append(f"&bull; <b>{path}</b>: {shown}<br>")

        if not lines:
            self.changed_label.setText(
                "<i>All snapshots use identical configuration parameters and no scans have been removed.</i>"
            )
            return
        self.changed_label.setText("".join(lines))

    # ---- tree -----------------------------------------------------------

    def _update_options_tree(self, snapshot: PMASnapshot) -> None:
        self.options_tree.clear()
        snap_idx = self._row_to_snapshot_index(self._current_row) if self._current_row is not None else None
        root = QTreeWidgetItem(self.options_tree, [f"snapshot[{snap_idx}]"])
        font = root.font(0)
        font.setBold(True)
        root.setFont(0, font)

        for name in _DATACLASS_OPTION_FIELDS:
            value = getattr(snapshot, name)
            child = QTreeWidgetItem(root, [name])
            child_font = child.font(0)
            child_font.setBold(True)
            child.setFont(0, child_font)
            self._add_dataclass_to_tree(child, value, prefix=name)

        for name in _SCALAR_OPTION_FIELDS:
            value = getattr(snapshot, name)
            leaf = QTreeWidgetItem(root, [name, _format_value(value)])
            leaf.setToolTip(0, name)
            self._maybe_highlight_leaf(leaf, name)

        self.options_tree.expandAll()

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
        for ax in (self.ax_horizontal, self.ax_vertical, self.ax_angles):
            ax.clear()
        self.canvas.draw()
        self.angles_canvas.draw()
        self.array_viewer.update_arrays(np.zeros((1, 1, 1), dtype=np.float32))
        self.changed_label.setText("<i>Sequence is empty.</i>")
        self.options_tree.clear()
        while self.snapshot_info_layout.rowCount():
            self.snapshot_info_layout.removeRow(0)


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
