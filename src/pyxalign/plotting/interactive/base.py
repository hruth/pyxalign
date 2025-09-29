from functools import wraps
from ctypes import ArgumentError
from operator import index
from re import L
from tkinter import Spinbox
from typing import Optional, Sequence, Callable, TypeVar, ParamSpec
import numpy as np
import cupy as cp
import traceback
import bisect
from pyxalign.api.options.plotting import ArrayViewerOptions
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLayout,
    QSlider,
    QLabel,
    QSpinBox,
    QPushButton,
    QApplication,
    QSpacerItem,
    QSizePolicy,
    QFrame,
    QCheckBox,
    QScrollArea,
)
from PyQt5.QtCore import (
    Qt,
    QTimer,
    QRunnable,
    pyqtSlot,
    QThreadPool,
    pyqtSignal,
    QObject,
    QEventLoop,
    QThread,
)
from PyQt5.QtGui import QPalette, QColor, QValidator

# Matplotlib imports remain here, but are not used in ArrayViewer anymore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# New import for using pyqtgraph
import pyqtgraph as pg

P = ParamSpec("P")
R = TypeVar("R")


class WorkerThread(QThread):
    finished = pyqtSignal(object)  # signal indicating multi_thread_func has completed

    def __init__(
        self, multi_thread_func: Callable[P, R], parent=None, *args: P.args, **kwargs: P.kwargs
    ):
        super().__init__(parent=parent)
        self.multi_thread_func = multi_thread_func
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        result = None
        result = self.multi_thread_func(*self.args, **self.kwargs)
        self.finished.emit(result)
        print("Thread complete")


class MultiThreadedWidget(QWidget):
    def __init__(self, multi_thread_func: Callable[P, R], parent=None):
        super().__init__(parent)
        self.multi_thread_func = multi_thread_func
        self.exception_received = False

    def start_thread(self, *args: P.args, **kwargs: P.kwargs) -> R | None:
        if not callable(self.multi_thread_func):
            raise ArgumentError("Cannot start multi-threading, valid function was not provided.")

        # create an event loop in the main thread
        self.event_loop = QEventLoop()
        # create the worker thread that will run the projection matching alignment
        self.worker_thread = WorkerThread(self.multi_thread_func, self, *args, **kwargs)
        # connect signals
        self.worker_thread.finished.connect(self._on_thread_finished)
        # start worker thread
        self.worker_thread.start()
        # block main thread until the worker finishes
        self.event_loop.exec_()
        self.worker_thread.wait()
        return self.thread_result

    def _on_thread_finished(self, result: R) -> None:
        self.thread_result = result
        self.event_loop.quit()


class ArrayViewer(MultiThreadedWidget):
    # Make array3d optional with a default of None
    def __init__(
        self,
        array3d: Optional[np.ndarray] = None,  # changed to Optional and given a default value
        options: Optional[ArrayViewerOptions] = None,
        sort_idx: Optional[Sequence] = None,
        multi_thread_func: Optional[Callable] = None,
        extra_title_strings_list: Optional[list[str]] = None,
        process_func: Optional[Callable] = None,
        parent=None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )
        self.extra_title_strings_list = extra_title_strings_list
        if process_func is None:
            if np.iscomplexobj(array3d):
                self.process_func = np.angle
            else:
                self.process_func = lambda x: x
        else:
            self.process_func = process_func

        # Only convert array if array3d is provided
        if array3d is not None:
            if cp.get_array_module(array3d) == cp:
                self.array3d = array3d.get()
            else:
                self.array3d = array3d
        else:
            # If no array is given, store None
            self.array3d = None

        self.sort_idx = sort_idx
        if options is None:
            self.options = ArrayViewerOptions()
        else:
            self.options = options

        # If array is given, use its shape. Otherwise, default to 1 to avoid errors.
        if self.array3d is not None:
            self.num_frames = self.array3d.shape[self.options.slider_axis]
        else:
            self.num_frames = 1

        self.playing = False

        # Create a pyqtgraph GraphicsLayoutWidget to hold the image
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.plot_item = self.graphics_layout.addPlot()
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)
        self.plot_item.setAspectLocked(True)

        # Create a checkbox for auto scaling the image intensities
        self.auto_clim_check_box = QCheckBox("Enable amplitude rescaling")
        self.auto_clim_check_box.setStyleSheet("QCheckBox {font-size: 15px;}")
        self.auto_clim_check_box.stateChanged.connect(self.refresh_frame)

        # Create index selection widget (slider, spinbox, play button, etc.)
        self.indexing_widget = IndexSelectorWidget(
            self.num_frames,
            self.options.start_index,
            additional_spinbox_indexing=self.options.additional_spinbox_indexing,
            additional_spinbox_title=self.options.additional_spinbox_titles,
            sort_idx=sort_idx,
            parent=parent,
        )
        self.slider, self.spinbox, self.play_button, self.timer = (
            self.indexing_widget.slider,
            self.indexing_widget.spinbox,
            self.indexing_widget.play_button,
            self.indexing_widget.play_timer,
        )
        self.slider.valueChanged.connect(self.update_frame)
        self.spinbox.valueChanged.connect(self.update_frame)
        self.play_button.clicked.connect(self.toggle_play)
        self.timer.timeout.connect(self.next_frame)

        # Main layout
        layout = QVBoxLayout()
        # In the original code, a matplotlib toolbar was added, but we omit it for pyqtgraph usage
        layout.addWidget(self.auto_clim_check_box)
        layout.addWidget(self.graphics_layout)
        layout.addWidget(self.indexing_widget)
        self.setLayout(layout)

        # Font and style adjustments
        small_style = """
            QWidget {
                font-size: 14px;
            }
            QSlider::groove:horizontal {
                height: 4px;
            }
            QSlider::handle:horizontal {
                width: 8px;
            }
        """
        self.setStyleSheet(small_style)

        # If array3d was provided, show the initial image
        if self.array3d is not None:
            self.display_frame(index=self.options.start_index)
            # force scaling
            self.image_item.setImage(autoLevels=True)

    def display_frame(self, index=0, force_autolim: bool = False):
        """Display a given slice (frame) from array3d."""
        # Only display if we have an array
        if self.array3d is None:
            self.plot_item.setTitle("No data loaded")
            return

        if self.sort_idx is not None:
            plot_index = self.sort_idx[index]
        else:
            plot_index = index

        image = self.array3d.take(indices=plot_index, axis=self.options.slider_axis)
        if cp.get_array_module(image) == cp:
            image = image.get()
        image = self.process_func(image)

        # Update pyqtgraph image
        if self.auto_clim_check_box.isChecked() or force_autolim:
            self.image_item.setImage(np.transpose(image), autoLevels=True)
        else:
            self.image_item.setImage(np.transpose(image), autoLevels=False)

        # Update title
        title = f"<span style='color:#B0E6F7'>Index {index}</span>"
        if self.extra_title_strings_list is not None:
            title += self.extra_title_strings_list[plot_index]
        title = "<b>" + title + "</b>"
        self.plot_item.setTitle(title)

    def update_frame(self, value, force_autolim: bool = False):
        self.display_frame(index=value, force_autolim=force_autolim)

    def toggle_play(self):
        if self.playing:
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start()
            self.play_button.setText("Pause")
        self.playing = not self.playing

    def next_frame(self):
        current = self.slider.value()
        next_idx = (current + 1) % self.num_frames
        self.slider.setValue(next_idx)

    def refresh_frame(self, force_autolim: bool = False):
        self.update_frame(self.slider.value(), force_autolim=force_autolim)

    def update_index_externally(self, index: int):
        self.slider.setValue(index)

    def reinitialize_all(
        self,
        array3d: Optional[np.ndarray] = None,
        sort_idx: Optional[Sequence] = None,
        extra_title_strings_list: Optional[Sequence] = None,
        new_additional_spinbox_indexing: Optional[list[np.ndarray]] = None,
        new_selected_value_list: Optional[list[int]] = None,
        process_func: Optional[Callable] = None,
    ):
        """Re-initialize the viewer with a new array or sort indices."""
        # Only process the new array if provided
        if array3d is not None:
            if process_func is None:
                if np.iscomplexobj(array3d):
                    self.process_func = np.angle
                else:
                    self.process_func = lambda x: x
            else:
                self.process_func = process_func

            self.array3d = array3d
            self.sort_idx = sort_idx
            self.extra_title_strings_list = extra_title_strings_list
            self.num_frames = self.array3d.shape[self.options.slider_axis]
            self.slider.setMaximum(self.num_frames - 1)
            self.spinbox.setMaximum(self.num_frames - 1)
            # refresh the frame
            self.refresh_frame(force_autolim=True)
            # update the other boxes
            new_selected_value_list = [
                arr[sort_idx[self.spinbox.value()]] for arr in new_additional_spinbox_indexing
            ]
            print(new_selected_value_list)
            self.indexing_widget.update_additional_spinbox_indexing(
                new_indexing=new_additional_spinbox_indexing,
                sort_idx=sort_idx,
                new_selected_value_list=new_selected_value_list,
            )

    def start(self):
        """Show the widget."""
        self.show()


class LinkedArrayViewer(MultiThreadedWidget):
    # Make array3d optional with a default of None
    def __init__(
        self,
        array_list: list[np.ndarray],
        options: Optional[ArrayViewerOptions] = None,
        sort_idx: Optional[Sequence] = None,
        multi_thread_func: Optional[Callable] = None,
        extra_title_strings_list: Optional[list[str]] = None,
        process_func: Optional[Callable] = None,
        parent=None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )
        # main layout
        layout = QHBoxLayout()
        # create array viewers
        self.array_viewer_list = []
        for i, array3d in enumerate(array_list):
            array_viewer = ArrayViewer(
                array3d,
                options=options,
                sort_idx=sort_idx,
                multi_thread_func=multi_thread_func,
                extra_title_strings_list=extra_title_strings_list,
                process_func=process_func,
            )
            self.array_viewer_list += [array_viewer]
            # connect all sliders
            if i != 0:
                array_viewer.slider.valueChanged.connect(self.array_viewer_list[0].slider.setValue)
                self.array_viewer_list[0].slider.valueChanged.connect(array_viewer.slider.setValue)
            layout.addWidget(array_viewer)
        self.setLayout(layout)


class IndexSelectorWidget(QWidget):
    def __init__(
        self,
        num_frames: int,
        start_index: Optional[int] = None,
        include_play_button: bool = True,
        additional_spinbox_indexing: Optional[list[np.ndarray]] = None,
        additional_spinbox_title: Optional[list[str]] = None,
        sort_idx: Optional[Sequence] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent=parent)
        self.extra_spinboxes_list: list[ValidatedSpinBox] = []
        if start_index is None:
            start_index = 0

        # Create a slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(num_frames - 1)
        self.slider.setValue(start_index)

        # SpinBox (editable + arrows)
        self.spinbox = ValidatedSpinBox(
            allowed_values=np.arange(0, num_frames, dtype=int)
        )  # QSpinBox()
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximum(num_frames - 1)
        self.spinbox.setValue(start_index)
        self.slider.valueChanged.connect(self.spinbox.setValue)
        self.spinbox.valueChanged.connect(self.slider.setValue)
        # add spinbox to layout with label
        main_spinbox_widget = QWidget()
        main_spinbox_widget.setLayout(QVBoxLayout())
        main_spinbox_widget.layout().setContentsMargins(
            0, *main_spinbox_widget.layout().getContentsMargins()[1:]
        )
        main_spinbox_widget.layout().addWidget(self.spinbox)
        main_spinbox_widget.layout().addWidget(QLabel("index"))

        # Play button
        if include_play_button:
            self.play_button = QPushButton("Play")
            self.play_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 3px 6px;    /* Inner spacing (top/bottom, left/right) */
                min-width: 60px;     /* Minimum width */
                min-height: 20px;    /* Minimum height */
                text-align: center;  /* Text alignment */
            }
            """)
            self.play_timer = QTimer()
            self.playback_speed_spin = QSpinBox()
            self.playback_speed_spin.setRange(1, 1000)  # Set a reasonable range
            self.playback_speed_spin.setValue(20)
            self.playback_speed_spin.valueChanged.connect(self._on_playback_speed_changed)
        else:
            self.play_button = QPushButton("Play")
            self.play_button.hide()

        # Layout for spinbox and optional play button
        spin_play_widget = QWidget()
        self.spin_play_layout = QHBoxLayout()
        self.spin_play_layout.setContentsMargins(0, *self.spin_play_layout.getContentsMargins()[1:])
        spin_play_widget.setLayout(self.spin_play_layout)
        if include_play_button:
            # A widget is made for the play button purely so I can
            # align it better on the gui
            play_button_widget = QWidget()
            play_button_widget.setLayout(QVBoxLayout())
            play_button_widget.layout().addWidget(self.play_button)
            play_button_widget.layout().addWidget(QLabel(" "))
            play_button_widget.layout().setContentsMargins(
                0, *play_button_widget.layout().getContentsMargins()[1:]
            )
            self.spin_play_layout.addWidget(play_button_widget, alignment=Qt.AlignTop)
        self.spin_play_layout.addWidget(main_spinbox_widget, alignment=Qt.AlignLeft | Qt.AlignTop)

        # add more spinboxes, like one for scan numbers in the case
        # of the ProjectionViewer, if specified
        if additional_spinbox_indexing is not None:
            for i, indexing in enumerate(additional_spinbox_indexing):
                sbox = ValidatedSpinBox(allowed_values=indexing[sort_idx])
                self.extra_spinboxes_list += [sbox]
                sbox.setValue(indexing[sort_idx[start_index]])
                sbox.setMinimum(np.min(indexing))
                sbox.setMaximum(np.max(indexing))

                # link to primary indexing box
                def update_extra_from_primary(i: int):
                    sbox.setValue(sbox.allowed_values[i])

                def update_primary_from_extra(i: int):
                    self.slider.setValue(np.where(np.array(sbox.allowed_values) == i)[0][0])

                self.slider.valueChanged.connect(update_extra_from_primary)
                sbox.valueChanged.connect(update_primary_from_extra)

                extra_sbox_widget = QWidget()
                extra_sbox_widget.setLayout(QVBoxLayout())
                extra_sbox_widget.layout().setContentsMargins(
                    0, *extra_sbox_widget.layout().getContentsMargins()[1:]
                )
                extra_sbox_widget.layout().addWidget(sbox)
                extra_sbox_widget.layout().addWidget(QLabel(additional_spinbox_title[i]))
                # add to spin-play layout
                self.spin_play_layout.addWidget(
                    extra_sbox_widget, alignment=Qt.AlignLeft | Qt.AlignTop
                )

        self.spin_play_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        if include_play_button:
            playback_speed_widget = QWidget()
            playback_speed_layout = QVBoxLayout()
            playback_speed_widget.setLayout(playback_speed_layout)

            playback_speed_label = QLabel("Playback Speed (Hz)")
            playback_speed_label.setStyleSheet("QLabel {font-size: 12px;}")
            playback_speed_layout.addWidget(self.playback_speed_spin)
            playback_speed_layout.addWidget(playback_speed_label)
            self.spin_play_layout.addWidget(
                playback_speed_widget, alignment=Qt.AlignRight | Qt.AlignTop
            )

        # Main layout for the index selector
        index_selection_layout = QVBoxLayout()
        index_selection_layout.addWidget(self.slider)
        index_selection_layout.addWidget(spin_play_widget)
        index_selection_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )
        self.setLayout(index_selection_layout)

        spin_play_widget.setStyleSheet("""
        QSpinBox {
            font-size: 14px;
            padding: 3px 6px;    /* Inner spacing (top/bottom, left/right) */
            min-width: 60px;     /* Minimum width */
            min-height: 20px;    /* Minimum height */
            text-align: center;  /* Text alignment */
        }
        """)
        # spin_play_widget.setStyleSheet("QLabel {font-size: 14px;}")

        if include_play_button:
            # Timer for playback
            self.play_timer = QTimer(parent)
            self.play_timer.setInterval(self.playback_speed_spin.value())  # milliseconds per frame

    def update_additional_spinbox_indexing(
        self,
        new_indexing: list[np.ndarray],
        sort_idx: Optional[np.ndarray] = None,
        new_selected_value_list: Optional[list[int]] = None,
    ):
        for i, indexing in enumerate(new_indexing):
            if sort_idx is not None:
                use_indexing = indexing[sort_idx]
            else:
                use_indexing = indexing
            if new_selected_value_list is None or new_selected_value_list[i] is None:
                new_value = None
            else:
                new_value = new_selected_value_list[i]
            self.extra_spinboxes_list[i].set_allowed_values(use_indexing, set_value_to=new_value)

    def _on_playback_speed_changed(self, value: int):
        interval = int(1e3 * 1 / value)
        self.play_timer.setInterval(interval)


class ValidatedSpinBox(QSpinBox):
    """
    QSpinBox that only accepts values from a discrete allowed set.

    - Preserves the input order of allowed_values (no sorting).
    - While typing, prefixes of allowed values are permitted (Intermediate).
    - Up/down arrows jump to the numerically closest next/previous allowed value.
    - allowed_values can be updated at runtime via set_allowed_values(...).
    - Optionally confirm only on Enter/focus-out via setKeyboardTracking(False).

    Note: this was primarily created by chatGPT5
    """

    def __init__(self, allowed_values, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_allowed_core(allowed_values)
        self.setRange(self._sorted_vals[0], self._sorted_vals[-1])
        self.setKeyboardTracking(False)  # confirm on Enter/focus-out (nice UX)

    # ---------- Public API ----------
    def set_allowed_values(
        self, allowed_values, set_value_to: Optional[int] = None
    ):  # *, reconcile="nearest"):
        """
        Update the allowed values after construction.

        Parameters
        ----------
        allowed_values : Iterable[int]
            New allowed integer values (deduplicated, preserving first occurrence).
        reconcile : {"nearest","min","max"}
            If current value not in new set:
              - "nearest": snap to numerically closest allowed value (default)
              - "min":     snap to the minimum of the new set
              - "max":     snap to the maximum of the new set
        """
        # if not allowed_values:
        #     raise ValueError("allowed_values must be non-empty")

        # cur = self.value()
        self._set_allowed_core(allowed_values)
        self.setRange(self._sorted_vals[0], self._sorted_vals[-1])

        if set_value_to is not None:
            self.setValue(set_value_to)

        # if cur not in self._set_vals:
        #     if reconcile == "min":
        #         target = self._sorted_vals[0]
        #     elif reconcile == "max":
        #         target = self._sorted_vals[-1]
        #     else:  # "nearest"
        #         target = self._nearest(cur)
        #     if target != self.value():
        #         self.setValue(target)

    def allowedValues(self):
        """Return the allowed values in the original input order."""
        return tuple(self.allowed_values)

    # ---------- Internals ----------
    def _set_allowed_core(self, allowed_values):
        # Deduplicate while preserving first occurrence (and cast to int)
        seen = set()
        vals = []
        for v in allowed_values:
            iv = int(v)
            if iv not in seen:
                seen.add(iv)
                vals.append(iv)
        if not vals:
            raise ValueError("allowed_values must be non-empty")

        self.allowed_values = vals  # original order (public)
        self._set_vals = set(vals)  # membership tests
        self._sorted_vals = sorted(vals)  # numeric stepping
        self._allowed_strs = [str(v) for v in vals]  # for prefix typing

    def _is_prefix_of_allowed(self, text: str) -> bool:
        s = text.strip()
        if s in ("", "+", "-"):
            return True
        neg = s.startswith("-")
        body = s[1:] if neg else s
        if not body.isdigit():
            return False
        # Allow leading zeros while typing
        s_no_zeros = ("-" if neg else "") + (body.lstrip("0") or "0")
        return any(asv.startswith(s) or asv.startswith(s_no_zeros) for asv in self._allowed_strs)

    def _nearest(self, x: int) -> int:
        """Return numerically closest value in _sorted_vals to x (ties -> smaller)."""
        a = self._sorted_vals
        i = bisect.bisect_left(a, x)
        if i == 0:
            return a[0]
        if i == len(a):
            return a[-1]
        before, after = a[i - 1], a[i]
        # tie-break toward the smaller (consistent, predictable)
        return before if (x - before) <= (after - x) else after

    # ---------- QAbstractSpinBox overrides ----------
    def validate(self, text, pos):
        try:
            val = int(text)
        except ValueError:
            return (
                QValidator.Intermediate if self._is_prefix_of_allowed(text) else QValidator.Invalid,
                text,
                pos,
            )

        if val in self._set_vals:
            return (QValidator.Acceptable, text, pos)

        if self._is_prefix_of_allowed(text):
            return (QValidator.Intermediate, text, pos)

        return (QValidator.Invalid, text, pos)

    def fixup(self, text):
        # On commit with invalid text, keep current valid value
        return str(self.value())

    def valueFromText(self, text):
        return int(text)

    def textFromValue(self, value):
        return str(value)

    def stepBy(self, steps):
        """
        Step to the numerically next/previous allowed value, regardless of
        input order. Supports multi-step (e.g., steps=±2).
        """
        if steps == 0:
            return

        cur = self.value()
        a = self._sorted_vals

        # If current value is not on-grid, snap to nearest first
        if cur not in self._set_vals:
            cur = self._nearest(cur)

        # Use bisect to find neighbors; loop for |steps| times
        for _ in range(abs(steps)):
            i = bisect.bisect_left(a, cur)
            if steps > 0:
                # move to strictly greater value if possible
                if i < len(a) and a[i] != cur:
                    cur = a[i]
                else:
                    # i points to current or end; go to next index if exists
                    next_i = min(i + 1, len(a) - 1)
                    cur = a[next_i]
            else:
                # move to strictly smaller value if possible
                if i > 0:
                    # a[i-1] is < cur (or == if off-grid handled above)
                    cur = a[i - 1] if (i >= len(a) or a[i] != cur) else a[i - 1]
                else:
                    cur = a[0]
        self.setValue(cur)
