from functools import wraps
from ctypes import ArgumentError
from typing import Optional, Sequence, Callable, TypeVar, ParamSpec
import numpy as np
import cupy as cp
import traceback
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
from PyQt5.QtGui import QPalette, QColor

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
    def __init__(
        self,
        array3d: np.ndarray,
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
            self.process_func = lambda x: x
        else:
            self.process_func = process_func

        # Convert cupy array to numpy array if needed
        if cp.get_array_module(array3d) == cp:
            self.array3d = array3d.get()
        else:
            self.array3d = array3d

        self.sort_idx = sort_idx
        if options is None:
            self.options = ArrayViewerOptions()
        else:
            self.options = options

        self.num_frames = self.array3d.shape[self.options.slider_axis]
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
            self.num_frames, self.options.start_index, parent=parent
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

        # Show the initial image
        self.display_frame(index=self.options.start_index)
        # force scaling
        self.image_item.setImage(autoLevels=True)

    def display_frame(self, index=0):
        """Display a given slice (frame) from array3d."""
        if self.sort_idx is not None:
            plot_index = self.sort_idx[index]
        else:
            plot_index = index
        image = self.array3d.take(indices=plot_index, axis=self.options.slider_axis)
        if cp.get_array_module(image) == cp:
            image = image.get()
        image = self.process_func(image)

        # Update pyqtgraph image
        # If auto scaling is enabled, let pyqtgraph handle levels automatically
        if self.auto_clim_check_box.isChecked():
            self.image_item.setImage(np.transpose(image), autoLevels=True)
        else:
            # Manually set levels from min/max of current frame
            self.image_item.setImage(np.transpose(image), autoLevels=False)
            # self.image_item.setLevels([image.min(), image.max()])

        # Update title
        title = f"Index {index}"
        if self.extra_title_strings_list is not None:
            title += self.extra_title_strings_list[plot_index]
        self.plot_item.setTitle(title)

    def update_frame(self, value):
        self.display_frame(index=value)

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

    def refresh_frame(self):
        self.update_frame(self.slider.value())

    def update_index_externally(self, index: int):
        self.slider.setValue(index)

    def reinitialize_all(
        self,
        array3d: np.ndarray,
        sort_idx: Optional[Sequence] = None,
        extra_title_strings_list: Optional[Sequence] = None,
    ):
        """Re-initialize the viewer with a new array or sort indices."""
        self.array3d = array3d
        self.sort_idx = sort_idx
        self.extra_title_strings_list = extra_title_strings_list
        self.num_frames = self.array3d.shape[self.options.slider_axis]
        self.slider.setMaximum(self.num_frames - 1)
        self.spinbox.setMaximum(self.num_frames - 1)
        self.refresh_frame()

    def start(self):
        """Show the widget."""
        self.show()


class IndexSelectorWidget(QWidget):
    def __init__(
        self,
        num_frames: int,
        start_index: Optional[int] = None,
        include_play_button: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent=parent)
        if start_index is None:
            start_index = 0

        # Create a slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(num_frames - 1)
        self.slider.setValue(start_index)

        # SpinBox (editable + arrows)
        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximum(num_frames - 1)
        self.spinbox.setValue(start_index)
        self.slider.valueChanged.connect(self.spinbox.setValue)
        self.spinbox.valueChanged.connect(self.slider.setValue)
        self.spinbox.setStyleSheet("""
        QSpinBox {
            font-size: 14px;
            padding: 3px 6px;    /* Inner spacing (top/bottom, left/right) */
            min-width: 60px;     /* Minimum width */
            min-height: 20px;    /* Minimum height */
            text-align: center;  /* Text alignment */
        }
        """)

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
            self.playback_speed_spin.setRange(10, 2000)  # Set a reasonable range
            self.playback_speed_spin.setValue(500)       # Default interval = 500ms
            self.playback_speed_spin.valueChanged.connect(self._on_playback_speed_changed)
        else:
            self.play_button = QPushButton("Play")
            self.play_button.hide()

        # Layout for spinbox and optional play button
        self.spin_play_layout = QHBoxLayout()
        if include_play_button:
            self.spin_play_layout.addWidget(self.play_button)
        self.spin_play_layout.addWidget(self.spinbox)
        self.spin_play_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        if include_play_button:
            self.spin_play_layout.addWidget(self.playback_speed_spin, alignment=Qt.AlignRight)

        # Main layout for the index selector
        index_selection_layout = QVBoxLayout()
        index_selection_layout.addWidget(self.slider)
        index_selection_layout.addLayout(self.spin_play_layout)
        index_selection_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )
        self.setLayout(index_selection_layout)

        # Timer for playback
        self.play_timer = QTimer(parent)
        self.play_timer.setInterval(self.playback_speed_spin.value())  # milliseconds per frame

    def _on_playback_speed_changed(self, value: int):
        self.play_timer.setInterval(value)

