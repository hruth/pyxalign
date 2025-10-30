from __future__ import annotations

from typing import Optional, List

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
)

from pyxalign.api.enums import RoundType
import pyxalign.data_structures.projections as p
from pyxalign.interactions.utils.loading_decorator import loading_bar_wrapper
from pyxalign.mask import place_patches_fourier_batch
from pyxalign.interactions.viewers.base import IndexSelectorWidget
from pyxalign.mask import clip_masks
from pyxalign.model_functions import symmetric_gaussian_2d
from pyxalign.transformations.helpers import round_to_divisor
from pyxalign.api import constants

"""
Interactive mask threshold selector based on pyqtgraph and the shared
IndexSelectorWidget used elsewhere in pyxalign.

This file replaces the previous Matplotlib-based implementation with a fast
Qt/pyqtgraph GUI and removes the bespoke slider / play logic in favour of the
centralised IndexSelectorWidget (see plotting/interactive/base.py).
"""


# ------------------------------------------------------------------------------
# Pyqtgraph canvas helper
# ------------------------------------------------------------------------------


class PGCanvas(QWidget):
    """
    Lightweight container holding three pyqtgraph ImageItems for:

      1. The binary mask
      2. mask * projection
      3. (1 - mask) * projection

    update_mask_plot(...) populates the images.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent=parent)

        self.graphics_layout = pg.GraphicsLayoutWidget()
        # 3 columns in a single row
        self.plot_mask = self.graphics_layout.addPlot(0, 0)
        self.plot_masked = self.graphics_layout.addPlot(0, 1)
        self.plot_inv = self.graphics_layout.addPlot(0, 2)

        for p in (self.plot_mask, self.plot_masked, self.plot_inv):
            p.setAspectLocked(True)

        self.img_mask = pg.ImageItem()
        self.img_masked = pg.ImageItem()
        self.img_inv = pg.ImageItem()

        self.plot_mask.addItem(self.img_mask)
        self.plot_masked.addItem(self.img_masked)
        self.plot_inv.addItem(self.img_inv)

        # Layout wrapper
        layout = QVBoxLayout()
        layout.addWidget(self.graphics_layout)
        self.setLayout(layout)

    # --------------------------------------------------------------------- #
    # Public API called from ThresholdSelector
    # --------------------------------------------------------------------- #

    def update_mask_plot(
        self,
        masks: np.ndarray,
        projections: np.ndarray,
        idx: int,
        threshold: float,
    ) -> None:
        """
        Update displayed images for frame *idx* and *threshold*.
        The logic matches the former Matplotlib implementation.
        """
        # build binary mask clip
        clipped_masks = (masks[idx] * 1).copy()
        clip_idx = clipped_masks > threshold
        clipped_masks[:] = 0
        clipped_masks[clip_idx] = 1

        # projection processing: amplitude or angle
        if np.isrealobj(projections[0, 0, 0]):

            def process_func(x):
                return x
        else:
            process_func = np.angle

        img_mask = clipped_masks
        img_masked = clipped_masks * process_func(projections[idx])
        img_inv = (1 - clipped_masks) * process_func(projections[idx])

        # pyqtgraph wants shape (rows, cols) – our arrays already comply
        self.img_mask.setImage(img_mask.T, autoLevels=True)
        self.img_masked.setImage(img_masked.T, autoLevels=True)
        self.img_inv.setImage(img_inv.T, autoLevels=True)

        # Titles
        self.plot_mask.setTitle("Mask")
        self.plot_masked.setTitle("Mask × Projection")
        self.plot_inv.setTitle("(1 - Mask) × Projection")


# ------------------------------------------------------------------------------
# Main interactive widget
# ------------------------------------------------------------------------------


class ThresholdSelector(QWidget):
    """
    Interactive tool to choose a binary-threshold for automatically
    generated “probe-patch” masks.

    Signals
    -------
    masks_created : np.ndarray
        Emitted once the user presses *Select and Finish*, containing the
        clipped/binary masks.
    """

    masks_created = pyqtSignal(np.ndarray)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        projections: "p.Projections",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent=parent)

        self.projections = projections
        self.options = self.projections.options.mask_from_positions

        # use simulated probe if specified by options; this typically
        # gives better results
        if self.options.use_simulated_probe:
            shape = self.projections.probe.shape
            probe_width = round_to_divisor(
                shape[0] * self.options.probe.fractional_width,
                round_type=RoundType.NEAREST,
                divisor=constants.divisor
            )
            probe = symmetric_gaussian_2d(shape, amplitude=1, sigma=probe_width)
        else:
            probe = self.projections.probe

        # Precompute masks (floating-point values)
        load_bar_func_wrapper = loading_bar_wrapper("Initializing masks...")(
            place_patches_fourier_batch
        )
        masks = load_bar_func_wrapper(
            self.projections.data.shape,
            probe,
            self.projections.probe_positions.data,
        )

        # Basic state
        self.masks = masks
        self.num_frames = len(masks)
        self.threshold = self.options.threshold
        self._playing = False

        # -------------------------------------------------------------- #
        # Visual components
        # -------------------------------------------------------------- #

        # Pyqtgraph canvas
        self.pg_canvas = PGCanvas(self)

        # Index selector (slider + spinbox + optional play)
        self.index_selector = IndexSelectorWidget(self.num_frames, 0, parent=self)
        self.slider = self.index_selector.slider
        self.spinbox = self.index_selector.spinbox
        self.play_button = self.index_selector.play_button
        self.timer = self.index_selector.play_timer  # QTimer from widget

        # Threshold spinbox
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1e9)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setValue(self.threshold)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)

        # Finish / accept button
        self.finish_button = QPushButton("Select and Finish")
        self.finish_button.setStyleSheet("background-color: blue; color: white;")
        self.finish_button.clicked.connect(self._finish)

        # -------------------------------------------------------------- #
        # Layout
        # -------------------------------------------------------------- #

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.pg_canvas)

        # threshold controls layout
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold"))
        thresh_layout.addWidget(self.threshold_spin)
        thresh_layout.addStretch()
        thresh_layout.addWidget(self.finish_button)
        main_layout.addLayout(thresh_layout)

        main_layout.addWidget(self.index_selector)

        # -------------------------------------------------------------- #
        # Signal wiring
        # -------------------------------------------------------------- #

        self.slider.valueChanged.connect(self._update_plot)
        self.spinbox.valueChanged.connect(self._update_plot)
        self.play_button.clicked.connect(self._toggle_play)
        self.timer.timeout.connect(self._next_frame)

        # Initial draw
        self._update_plot(initial=True)

        # Window properties
        self.setWindowTitle("Threshold Selector")
        self.resize(1000, 650)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _on_threshold_changed(self) -> None:
        self.threshold = self.threshold_spin.value()
        self._update_plot()

    def _update_plot(self, value: int | None = None, *, initial: bool = False) -> None:
        """
        Refresh pyqtgraph images. Called when slider/spinbox changes or
        threshold changes.
        """
        if value is None:  # called from threshold change
            value = self.slider.value()
        self.pg_canvas.update_mask_plot(
            self.masks,
            self.projections.data,
            idx=value,
            threshold=self.threshold,
        )
        # Force initial autoscale to avoid first-frame low contrast
        if initial:
            for img in (self.pg_canvas.img_mask, self.pg_canvas.img_masked, self.pg_canvas.img_inv):
                img.setImage(img.image, autoLevels=True)

    # ------------------------------------------------------------------ #
    # Playback helpers
    # ------------------------------------------------------------------ #

    def _toggle_play(self) -> None:
        if self._playing:
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start()
            self.play_button.setText("Pause")
        self._playing = not self._playing

    def _next_frame(self) -> None:
        current = self.slider.value()
        next_idx = (current + 1) % self.num_frames
        self.slider.setValue(next_idx)

    # ------------------------------------------------------------------ #
    # Finish & emit
    # ------------------------------------------------------------------ #

    def _finish(self) -> None:
        """Disable UI, clip masks, emit signal, close window."""
        # Stop any playback
        self.timer.stop()

        # Convert masks to binary using final threshold
        wrapped_clip_masks = loading_bar_wrapper("Constructing masks...")(func=clip_masks)
        self.masks = wrapped_clip_masks(self.masks, self.threshold)

        # update Projection object masks
        new_masks = self.masks
        if self.projections.masks is None or self.projections.masks.shape != new_masks.shape:
            self.projections.masks = new_masks
        else:
            self.projections.masks[:] = new_masks
        # update threshold in projection options
        self.projections.options.mask_from_positions.threshold = self.threshold

        # Emit and close
        self.masks_created.emit(self.masks)
        self.close()


def launch_mask_builder(
    projections: "p.Projections",
    wait_until_closed: bool = False,
):
    app = QApplication.instance() or QApplication([])
    gui = ThresholdSelector(projections)
    gui.show()
    gui.setAttribute(Qt.WA_DeleteOnClose)
    if wait_until_closed:
        app.exec_()
    return gui
