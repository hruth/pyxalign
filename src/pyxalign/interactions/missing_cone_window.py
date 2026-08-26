"""
Interactive GUI for running the fill_missing_cone reconstruction regularization.

This module provides a two-tab window for:
- Configuring and running fill_missing_cone with live volume preview.
- Browsing and comparing results from previous runs.
"""

import copy
from datetime import datetime
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyxalign.api.options.missing_cone import FillMissingConeOptions
from pyxalign.api.options.plotting import ArrayViewerOptions
from pyxalign.api.options.transform import Crop3DOptions
from pyxalign.interactions.crop_3d_selector import GetCrop3DOptionsFromSelector
from pyxalign.interactions.histogram_viewer import VolumeHistogramViewer
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.utils.loading_display_tools import loading_bar_wrapper
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
from pyxalign.interactions.viewers.base import ArrayViewer
from pyxalign.interactions.viewers.utils import OptionsDisplayWidget
from pyxalign.missing_cone import fill_missing_cone
from pyxalign.transformations.classes import Cropper3D


_COL_RUN = 0
_COL_TIMESTAMP = 1
_HISTORY_HEADERS = ["Run #", "Timestamp"]


class _VolumeViewerPanel(QWidget):
    """Single ArrayViewer panel for displaying a 3D volume.

    Uses a single ArrayViewer with axis-cycling controls visible
    (``hide_axis_controls=False``) and volume saving enabled.  When
    ``update_volume`` is called, the existing viewer is updated in-place via
    ``reinitialize_all`` so that widget state (zoom, colour limits) is
    preserved as much as possible.  The slider index is kept when the new
    volume has the same size along the currently displayed axis.
    """

    def __init__(self, volume: Optional[np.ndarray], parent=None):
        super().__init__(parent)
        self._inner_layout = QVBoxLayout(self)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._viewer: Optional[ArrayViewer] = None

        if volume is not None:
            self._build_viewer(volume)
        else:
            self._show_placeholder()

    def _show_placeholder(self):
        self._clear_layout()
        label = QLabel("No volume to display.\nSelect a run entry from the table.")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("QLabel { color: gray; font-size: 14px; }")
        self._inner_layout.addWidget(label)

    def _clear_layout(self):
        while self._inner_layout.count():
            item = self._inner_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_viewer(self, volume: np.ndarray):
        """Create a fresh ArrayViewer widget for the given volume."""
        self._clear_layout()
        self._viewer = ArrayViewer(
            array3d=volume,
            options=ArrayViewerOptions(
                slider_axis=0,
                start_index=volume.shape[0] // 2,
            ),
            hide_axis_controls=False,
            include_array_saving_widget=True,
        )
        self._inner_layout.addWidget(self._viewer)

    def update_volume(self, volume: Optional[np.ndarray]):
        """Replace the displayed volume.

        Uses ``reinitialize_all`` on the existing viewer to avoid rebuilding
        the widget.  The slider index is preserved when the new volume has the
        same size along the currently displayed axis; otherwise it resets to
        the centre.
        """
        if volume is None:
            self._viewer = None
            self._show_placeholder()
            return

        if self._viewer is None:
            # First non-None volume: create the widget from scratch.
            self._build_viewer(volume)
            return

        # Capture state before reinitializing so we can restore it.
        current_axis = self._viewer.options.slider_axis
        current_index = self._viewer.slider.value()
        old_axis_size = (
            self._viewer.array3d.shape[current_axis]
            if self._viewer.array3d is not None
            else 0
        )

        # reinitialize_all updates num_frames, clamps the slider range, and
        # calls refresh_frame — matching the pattern used in PMASequenceViewer.
        self._viewer.reinitialize_all(array3d=volume)

        # Restore the absolute index when the axis length is unchanged.
        if volume.shape[current_axis] == old_axis_size:
            self._viewer.slider.setValue(current_index)


def _crops_are_equal(a: Optional[Crop3DOptions], b: Optional[Crop3DOptions]) -> bool:
    """Return True if two Crop3DOptions instances have identical crop geometry."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return (
        a.enabled == b.enabled
        and a.horizontal_range == b.horizontal_range
        and a.vertical_range == b.vertical_range
        and a.depth_range == b.depth_range
        and a.horizontal_offset == b.horizontal_offset
        and a.vertical_offset == b.vertical_offset
        and a.depth_offset == b.depth_offset
    )


def _crop_summary_string(crop_options: Crop3DOptions) -> str:
    """Return a compact human-readable summary of Crop3DOptions."""
    if not crop_options.enabled:
        return "disabled"
    parts = []
    if crop_options.depth_range is not None:
        parts.append(f"D:{crop_options.depth_range}+{crop_options.depth_offset}")
    if crop_options.vertical_range is not None:
        parts.append(f"V:{crop_options.vertical_range}+{crop_options.vertical_offset}")
    if crop_options.horizontal_range is not None:
        parts.append(f"H:{crop_options.horizontal_range}+{crop_options.horizontal_offset}")
    return ", ".join(parts) if parts else "full range"


class FillMissingConeWindow(QWidget):
    """Interactive window for running and inspecting fill_missing_cone results.

    The window has two tabs:

    **Run tab** - Configure laminography angle and FillMissingConeOptions, optionally
    select a 3D crop region, then run the algorithm.  The result is displayed in a
    volume viewer (with saving and axis-cycling enabled) on the right.

    **History tab** - Every completed run is recorded in a table (Run #, Timestamp).
    Selecting a row shows the laminography angle and full FillMissingConeOptions for
    that run below the table, and displays the corresponding result volume on the right.
    Individual volumes can be deleted from memory while retaining their table entry.

    Args:
        volume: The 3D reconstruction to regularize.
        lamino_angle: Initial laminography angle in degrees.
        options: FillMissingConeOptions instance whose fields are updated live by
            the GUI.  Pass a shared instance to propagate changes to the caller.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        volume: np.ndarray,
        lamino_angle: float,
        options: FillMissingConeOptions,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.input_volume = volume
        self.lamino_angle = lamino_angle
        self.options = options
        self.crop_3d_selector_window: Optional[QWidget] = None
        self._histogram_window: Optional[VolumeHistogramViewer] = None
        self._comparison_window: Optional[QWidget] = None
        self._comparison_viewer: Optional[ArrayViewer] = None
        self._comparison_crop_3d: Optional[Crop3DOptions] = None
        self._comparison_window_header: Optional[QLabel] = None
        self._last_selected_row: int = -1
        self._is_syncing: bool = False

        # Each entry: dict with keys run_number, timestamp, lamino_angle,
        # options (deep-copied FillMissingConeOptions), volume (ndarray or None)
        self._history: list[dict] = []

        self._setup_ui()
        self.setWindowTitle("Fill Missing Cone")
        self.resize(1200, 850)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Build the top-level tab widget."""
        tabs = QTabWidget()
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(tabs)

        self._run_tab_viewer: Optional[_VolumeViewerPanel] = None
        self._history_tab_viewer: Optional[_VolumeViewerPanel] = None
        self._history_table: Optional[QTableWidget] = None
        self._history_lamino_label: Optional[QLabel] = None
        self._history_options_display: Optional[OptionsDisplayWidget] = None
        self._compare_button: Optional[QPushButton] = None

        tabs.addTab(self._create_run_tab(), "Run")
        tabs.addTab(self._create_history_tab(), "History")

    def _create_run_tab(self) -> QWidget:
        """Build the Run tab: controls on the left, volume viewer on the right."""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        layout.addWidget(self._create_run_controls(), stretch=1)

        self._run_tab_viewer = _VolumeViewerPanel(self.input_volume)
        layout.addWidget(self._run_tab_viewer, stretch=3)

        return tab

    def _create_run_controls(self) -> QWidget:
        """Build the left-hand control panel for the Run tab."""
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)

        panel_layout.addWidget(self._create_lamino_angle_group())
        panel_layout.addWidget(self._create_crop_button())
        panel_layout.addWidget(self._create_histogram_button())
        panel_layout.addWidget(self._create_options_editor())
        panel_layout.addWidget(self._create_run_button())
        panel_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(scroll)
        return container

    def _create_lamino_angle_group(self) -> QGroupBox:
        group = QGroupBox("Laminography Angle")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Angle (deg):"))
        self._lamino_angle_spinbox = QDoubleSpinBox()
        self._lamino_angle_spinbox.setRange(0.0, 90.0)
        self._lamino_angle_spinbox.setDecimals(4)
        self._lamino_angle_spinbox.setSingleStep(0.1)
        self._lamino_angle_spinbox.setValue(self.lamino_angle)
        self._lamino_angle_spinbox.valueChanged.connect(self._on_lamino_angle_changed)
        layout.addWidget(self._lamino_angle_spinbox)
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        return group

    def _create_crop_button(self) -> QPushButton:
        self._select_crop_button = QPushButton("Select 3D Crop")
        self._select_crop_button.clicked.connect(self._open_crop_3d_selector)
        return self._select_crop_button

    def _create_histogram_button(self) -> QPushButton:
        self._histogram_button = QPushButton("Inspect Histogram of Input Volume")
        self._histogram_button.clicked.connect(self._open_histogram_viewer)
        return self._histogram_button

    def _create_options_editor(self) -> BasicOptionsEditor:
        self._options_editor = BasicOptionsEditor(
            data=self.options,
            skip_fields=["crop_3d"],
            label="Fill Missing Cone Options",
        )
        return self._options_editor

    def _create_run_button(self) -> QPushButton:
        self._run_button = QPushButton("Run")
        self._run_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; font-size: 14px; padding: 10px; }"
            "QPushButton:pressed { background-color: #388E3C; }"
        )
        self._run_button.clicked.connect(self._run_fill_missing_cone)
        return self._run_button

    def _create_history_tab(self) -> QWidget:
        """Build the History tab: run table on the left, volume viewer on the right."""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        layout.addWidget(self._create_history_controls(), stretch=1)

        self._history_tab_viewer = _VolumeViewerPanel(None)
        layout.addWidget(self._history_tab_viewer, stretch=3)

        return tab

    def _create_history_controls(self) -> QWidget:
        """Build the left panel for the History tab."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        table_title = QLabel("Run History:")
        table_title.setStyleSheet("QLabel { font-size: 16px; }")
        layout.addWidget(table_title)

        self._history_table = QTableWidget()
        self._history_table.setColumnCount(len(_HISTORY_HEADERS))
        self._history_table.setHorizontalHeaderLabels(_HISTORY_HEADERS)
        self._history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._history_table.setSelectionMode(QTableWidget.SingleSelection)
        self._history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._history_table.verticalHeader().setVisible(False)
        self._history_table.itemSelectionChanged.connect(self._on_history_row_selected)
        layout.addWidget(self._history_table)

        self._delete_volume_button = QPushButton("Delete Volume for Selected Entry")
        self._delete_volume_button.setStyleSheet(
            "QPushButton { background-color: #CC3333; color: white; "
            "font-weight: bold; padding: 6px; }"
            "QPushButton:pressed { background-color: #991111; }"
            "QPushButton:disabled { background-color: #AAAAAA; }"
        )
        self._delete_volume_button.setEnabled(False)
        self._delete_volume_button.clicked.connect(self._delete_selected_volume)
        layout.addWidget(self._delete_volume_button)

        self._compare_button = QPushButton("Compare with Original Volume")
        self._compare_button.setEnabled(False)
        self._compare_button.clicked.connect(self._open_comparison_window)
        layout.addWidget(self._compare_button)

        config_title = QLabel("Run Configuration:")
        config_title.setStyleSheet("QLabel { font-size: 16px; }")
        layout.addWidget(config_title)

        self._history_lamino_label = QLabel("Laminography Angle: —")
        self._history_lamino_label.setStyleSheet("QLabel { font-size: 13px; }")
        layout.addWidget(self._history_lamino_label)

        self._history_options_display = OptionsDisplayWidget()
        layout.addWidget(self._history_options_display)

        return panel

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_lamino_angle_changed(self, value: float):
        self.lamino_angle = value

    def _open_histogram_viewer(self):
        """Open the VolumeHistogramViewer for the input volume, reusing any open window."""
        try:
            if self._histogram_window is not None and self._histogram_window.isVisible():
                self._histogram_window.raise_()
                self._histogram_window.activateWindow()
                return
        except RuntimeError:
            self._histogram_window = None

        self._histogram_window = VolumeHistogramViewer(self.input_volume)
        self._histogram_window.setAttribute(Qt.WA_DeleteOnClose)
        self._histogram_window.show()

    def _open_crop_3d_selector(self):
        """Open the interactive 3D crop selector window."""
        try:
            if self.crop_3d_selector_window is not None:
                self.crop_3d_selector_window.isVisible()
        except RuntimeError:
            self.crop_3d_selector_window = None

        self.crop_3d_selector_window = GetCrop3DOptionsFromSelector(
            array3d=self.input_volume,
            crop_options=self.options.crop_3d,
        )
        self.crop_3d_selector_window.crop_3d_selected.connect(self._on_crop_selected)
        self.crop_3d_selector_window.show()

    def _on_crop_selected(self):
        """Apply the crop selection back to the options instance."""
        if self.crop_3d_selector_window is not None:
            self.options.crop_3d = self.crop_3d_selector_window.options
            self.crop_3d_selector_window.close()

    def _run_fill_missing_cone(self):
        """Apply crop (if enabled), run fill_missing_cone, update viewer and history."""
        options_copy = copy.deepcopy(self.options)
        lamino_angle_snapshot = self.lamino_angle

        cropper = Cropper3D(options=copy.deepcopy(self.options.crop_3d))
        volume_to_process = cropper.run(self.input_volume)

        result_holder = [None]

        def _run():
            result_holder[0] = fill_missing_cone(
                rec=volume_to_process,
                lamino_angle=lamino_angle_snapshot,
                delta_background=options_copy.delta_background,
                delta_maximal=options_copy.delta_maximal,
                mask_relax=options_copy.mask_relax,
                max_scale=options_copy.max_scale,
                n_iter=options_copy.n_iter,
                tv_lambda=options_copy.tv_lambda,
            )

        run_wrapped = loading_bar_wrapper("Running fill_missing_cone...")(_run)
        run_wrapped()

        result_volume = result_holder[0]
        if result_volume is None:
            QMessageBox.critical(self, "Error", "fill_missing_cone returned no result.")
            return

        self._run_tab_viewer.update_volume(result_volume)

        self._record_history_entry(
            lamino_angle=lamino_angle_snapshot,
            options=options_copy,
            volume=result_volume,
        )

    def _record_history_entry(
        self,
        lamino_angle: float,
        options: FillMissingConeOptions,
        volume: np.ndarray,
    ):
        """Add a completed run to the history list and table."""
        run_number = len(self._history) + 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            "run_number": run_number,
            "timestamp": timestamp,
            "lamino_angle": lamino_angle,
            "options": options,
            "volume": volume,
        }
        self._history.append(entry)

        row = self._history_table.rowCount()
        self._history_table.insertRow(row)
        self._history_table.setItem(row, _COL_RUN, QTableWidgetItem(str(run_number)))
        self._history_table.setItem(row, _COL_TIMESTAMP, QTableWidgetItem(timestamp))

        # Auto-select the new row so the history display updates immediately
        self._history_table.selectRow(row)

    def _on_history_row_selected(self):
        """Update the config display and volume viewer when a table row is selected.

        Enforces a single persistent selection: if all rows are deselected while
        entries exist, the previous selection is silently restored.  When the row
        changes and a comparison window is open, the comparison is updated rather
        than closed (unless the new entry has no volume).
        """
        selected_rows = self._history_table.selectionModel().selectedRows()

        # Enforce that exactly one row is always selected when entries exist.
        if not selected_rows:
            if self._history_table.rowCount() > 0:
                restore_row = max(0, self._last_selected_row)
                self._history_table.selectionModel().blockSignals(True)
                self._history_table.selectRow(restore_row)
                self._history_table.selectionModel().blockSignals(False)
            else:
                self._delete_volume_button.setEnabled(False)
                self._compare_button.setEnabled(False)
            return

        row = selected_rows[0].row()
        entry = self._history[row]

        # Handle comparison window when the selected row changes.
        if row != self._last_selected_row and self._comparison_window is not None:
            if entry["volume"] is not None:
                new_crop = entry["options"].crop_3d
                if not _crops_are_equal(new_crop, self._comparison_crop_3d):
                    self._update_comparison_for_entry(entry)
                # Same crop → leave the comparison window unchanged.
            else:
                # No volume to compare against; close the window.
                self._close_comparison_window()

        self._last_selected_row = row

        has_volume = entry["volume"] is not None
        self._delete_volume_button.setEnabled(True)
        self._compare_button.setEnabled(has_volume)

        self._history_lamino_label.setText(
            f"Laminography Angle: {entry['lamino_angle']:.4f}°"
        )
        self._history_options_display.update_options(entry["options"])
        self._history_options_display.update_display()

        self._history_tab_viewer.update_volume(entry["volume"])

    def _delete_selected_volume(self):
        """Free the volume array for the selected history entry to save memory."""
        selected_rows = self._history_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        entry = self._history[row]

        if entry["volume"] is None:
            QMessageBox.information(
                self,
                "Already deleted",
                f"The volume for run #{entry['run_number']} has already been deleted.",
            )
            return

        confirm = QMessageBox.question(
            self,
            "Delete Volume",
            f"Delete the stored volume for run #{entry['run_number']}?\n"
            "This will free GPU/CPU memory but cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        entry["volume"] = None
        self._close_comparison_window()
        self._compare_button.setEnabled(False)
        self._history_tab_viewer.update_volume(None)

        # Gray out the table row to indicate the volume is gone
        for col in range(self._history_table.columnCount()):
            item = self._history_table.item(row, col)
            if item is not None:
                item.setForeground(Qt.gray)


    def _close_comparison_window(self):
        """Close and discard any open comparison window."""
        if self._comparison_window is None:
            return
        try:
            self._comparison_window.close()
        except RuntimeError:
            pass
        self._comparison_window = None
        self._comparison_viewer = None
        self._comparison_crop_3d = None
        self._comparison_window_header = None

    def _open_comparison_window(self):
        """Open a linked viewer showing the cropped original volume for the selected run."""
        selected_rows = self._history_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        entry = self._history[row]

        if entry["volume"] is None:
            return

        self._close_comparison_window()

        # Apply the saved crop with return_view=True so no extra memory is used.
        crop_opts = copy.deepcopy(entry["options"].crop_3d)
        crop_opts.return_view = True
        cropped_original = Cropper3D(options=crop_opts).run(self.input_volume)

        # Match the history viewer's current axis and index so the windows open
        # in sync without any user interaction.
        history_viewer = self._history_tab_viewer._viewer
        start_axis = history_viewer.options.slider_axis if history_viewer is not None else 0
        start_index = (
            min(history_viewer.slider.value(), cropped_original.shape[start_axis] - 1)
            if history_viewer is not None
            else cropped_original.shape[start_axis] // 2
        )

        comparison_viewer = ArrayViewer(
            array3d=cropped_original,
            options=ArrayViewerOptions(
                slider_axis=start_axis,
                start_index=start_index,
            ),
            hide_axis_controls=False,
            include_array_saving_widget=True,
        )

        crop_summary = _crop_summary_string(entry["options"].crop_3d)
        title = f"Original Volume — Run #{entry['run_number']} (crop: {crop_summary})"

        header = QLabel(title)
        header.setStyleSheet("QLabel { font-size: 13px; font-weight: bold; }")

        window = QWidget()
        window.setAttribute(Qt.WA_DeleteOnClose)
        window.setWindowTitle(title)
        layout = QVBoxLayout(window)
        layout.addWidget(header)
        layout.addWidget(comparison_viewer)
        window.resize(700, 600)
        window.destroyed.connect(self._on_comparison_window_destroyed)
        window.show()

        self._comparison_window = window
        self._comparison_viewer = comparison_viewer
        self._comparison_crop_3d = entry["options"].crop_3d
        self._comparison_window_header = header

        if history_viewer is not None:
            self._connect_comparison_signals(history_viewer, comparison_viewer)

    def _on_comparison_window_destroyed(self):
        """Clear comparison references when the window is closed by the user."""
        self._comparison_window = None
        self._comparison_viewer = None
        self._comparison_crop_3d = None
        self._comparison_window_header = None

    def _update_comparison_for_entry(self, entry: dict):
        """Update the open comparison window to reflect a new entry's crop."""
        crop_opts = copy.deepcopy(entry["options"].crop_3d)
        crop_opts.return_view = True
        new_cropped = Cropper3D(options=crop_opts).run(self.input_volume)

        crop_summary = _crop_summary_string(entry["options"].crop_3d)
        title = f"Original Volume — Run #{entry['run_number']} (crop: {crop_summary})"
        self._comparison_window.setWindowTitle(title)
        self._comparison_window_header.setText(title)

        # Block the sync flag so the slider changes during reinitialize_all
        # don't cascade back to the history viewer.
        self._is_syncing = True
        self._comparison_viewer.reinitialize_all(array3d=new_cropped)
        history_viewer = self._history_tab_viewer._viewer
        if history_viewer is not None:
            clamped = min(
                history_viewer.slider.value(), self._comparison_viewer.slider.maximum()
            )
            self._comparison_viewer.slider.setValue(clamped)
        self._is_syncing = False

        self._comparison_crop_3d = entry["options"].crop_3d

    # ------------------------------------------------------------------
    # Viewer sync — instance methods so there are no closure GC issues and
    # _is_syncing is always reset via try/finally even on exceptions.
    # Both sides look up the current viewer objects dynamically.
    # ------------------------------------------------------------------

    def _sync_index_to_comparison(self, value: int):
        """Forward history viewer slider change to the comparison viewer."""
        if self._is_syncing or self._comparison_viewer is None:
            return
        self._is_syncing = True
        try:
            clamped = min(value, self._comparison_viewer.slider.maximum())
            self._comparison_viewer.slider.setValue(clamped)
        except RuntimeError:
            self._close_comparison_window()
        finally:
            self._is_syncing = False

    def _sync_index_to_history(self, value: int):
        """Forward comparison viewer slider change to the history viewer."""
        if self._is_syncing:
            return
        history_viewer = self._history_tab_viewer._viewer
        if history_viewer is None:
            return
        self._is_syncing = True
        try:
            clamped = min(value, history_viewer.slider.maximum())
            history_viewer.slider.setValue(clamped)
        except RuntimeError:
            pass
        finally:
            self._is_syncing = False

    def _sync_axis_to_comparison(self):
        """Forward history viewer axis change to the comparison viewer.

        Called via history viewer axis buttons' ``clicked`` signal, which fires
        *after* ``cycle_axis_forward/backward`` has already updated
        ``options.slider_axis``.
        """
        if self._is_syncing or self._comparison_viewer is None:
            return
        history_viewer = self._history_tab_viewer._viewer
        if history_viewer is None:
            return
        self._is_syncing = True
        try:
            new_axis = history_viewer.options.slider_axis
            self._comparison_viewer._update_axis_memory(self._comparison_viewer.slider.value())
            self._comparison_viewer.options.slider_axis = new_axis
            self._comparison_viewer._update_after_axis_change()
            clamped = min(history_viewer.slider.value(), self._comparison_viewer.slider.maximum())
            self._comparison_viewer.slider.setValue(clamped)
        except RuntimeError:
            self._close_comparison_window()
        finally:
            self._is_syncing = False

    def _sync_axis_to_history(self):
        """Forward comparison viewer axis change to the history viewer."""
        if self._is_syncing or self._comparison_viewer is None:
            return
        history_viewer = self._history_tab_viewer._viewer
        if history_viewer is None:
            return
        self._is_syncing = True
        try:
            new_axis = self._comparison_viewer.options.slider_axis
            history_viewer._update_axis_memory(history_viewer.slider.value())
            history_viewer.options.slider_axis = new_axis
            history_viewer._update_after_axis_change()
            clamped = min(self._comparison_viewer.slider.value(), history_viewer.slider.maximum())
            history_viewer.slider.setValue(clamped)
        except RuntimeError:
            pass
        finally:
            self._is_syncing = False

    def _connect_comparison_signals(
        self, history_viewer: ArrayViewer, comparison_viewer: ArrayViewer
    ):
        """Wire up bidirectional slider and axis sync between the two viewers.

        Disconnects any pre-existing connections from *history_viewer* to the
        sync methods before connecting, so reopening the comparison window
        never creates duplicate connections.
        """
        for sig in (
            history_viewer.slider.valueChanged,
            history_viewer.next_axis_button.clicked,
            history_viewer.prev_axis_button.clicked,
        ):
            try:
                sig.disconnect(self._sync_index_to_comparison)
            except (TypeError, RuntimeError):
                pass
            try:
                sig.disconnect(self._sync_axis_to_comparison)
            except (TypeError, RuntimeError):
                pass

        history_viewer.slider.valueChanged.connect(self._sync_index_to_comparison)
        history_viewer.next_axis_button.clicked.connect(self._sync_axis_to_comparison)
        history_viewer.prev_axis_button.clicked.connect(self._sync_axis_to_comparison)

        comparison_viewer.slider.valueChanged.connect(self._sync_index_to_history)
        comparison_viewer.next_axis_button.clicked.connect(self._sync_axis_to_history)
        comparison_viewer.prev_axis_button.clicked.connect(self._sync_axis_to_history)


@switch_to_matplotlib_qt_backend
def launch_fill_missing_cone_gui(
    volume: np.ndarray,
    lamino_angle: float,
    options: Optional[FillMissingConeOptions] = None,
    wait_until_closed: bool = False,
) -> FillMissingConeWindow:
    """Launch an interactive GUI for running and inspecting fill_missing_cone results.

    Args:
        volume: 3D reconstruction array to regularize.
        lamino_angle: Laminography angle in degrees (editable in the GUI).
        options: FillMissingConeOptions instance.  If None, a default instance is
            created.  The GUI updates this instance in-place so callers can read
            back modified values after the window is used.
        wait_until_closed: If True, blocks until the window is closed.

    Returns:
        The FillMissingConeWindow widget.

    Example:
        Launch the GUI with a reconstructed volume::

            options = FillMissingConeOptions()
            gui = pyxalign.gui.launch_fill_missing_cone_gui(
                volume=volume_3d,
                lamino_angle=79.0,
                options=options,
            )
    """
    if options is None:
        options = FillMissingConeOptions()

    app = QApplication.instance() or QApplication([])
    gui = FillMissingConeWindow(volume=volume, lamino_angle=lamino_angle, options=options)
    gui.show()
    if wait_until_closed:
        app.exec_()
    return gui
