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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
import threading

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
        parent=None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )
        self.extra_title_strings_list = extra_title_strings_list

        if cp.get_array_module(array3d) == cp:
            self.array3d = array3d.get()
        else:
            self.array3d = array3d

        self.sort_idx = sort_idx
        if options is None:
            self.options = ArrayViewerOptions()
        else:
            self.options = options
        self.num_frames = array3d.shape[self.options.slider_axis]
        self.playing = False

        # Create the figure, canvas, and initial image
        self.figure = Figure(layout="compressed")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setFixedHeight(30)
        self.ax = self.figure.add_subplot(111)
        self.im = self.ax.imshow(
            self.array3d.take(
                indices=self.options.start_index,
                axis=self.options.slider_axis,
            ),
            cmap="bone",
        )
        self.canvas.draw()

        self.slider, self.spinbox, self.play_button, self.timer, index_selection_layout = (
            return_spin_play_slider_widgets(self.num_frames, self.options.start_index)
        )
        self.slider.valueChanged.connect(self.update_frame)
        self.spinbox.valueChanged.connect(self.update_frame)
        self.play_button.clicked.connect(self.toggle_play)
        self.timer.timeout.connect(self.next_frame)

        # Radio button for toggling auto clim adjustment
        self.auto_clim_check_box = QCheckBox("Enable amplitude rescaling")
        self.auto_clim_check_box.stateChanged.connect(self.refresh_frame)
        self.auto_clim_check_box.setStyleSheet("QCheckBox {font-size: 15px;}")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.auto_clim_check_box)
        layout.addWidget(self.canvas)
        layout.addLayout(index_selection_layout)
        self.setLayout(layout)

        # was 10
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

        # Refresh the display at the end of the window initialization
        self.display_frame(index=self.options.start_index)

    def display_frame(self, index=0):
        if self.sort_idx is not None:
            plot_index = self.sort_idx[index]
        else:
            plot_index = index
        image = self.array3d.take(indices=plot_index, axis=self.options.slider_axis)
        if cp.get_array_module(image) == cp:
            image = image.get()
        self.im.set_data(image)  # faster than the clear() and imshow() method
        title = f"Index {index}"
        if self.extra_title_strings_list is not None:
            title += self.extra_title_strings_list[plot_index]
        self.ax.set_title(title)
        if self.auto_clim_check_box.isChecked():
            self.im.autoscale()
        self.canvas.draw_idle()  # faster than the draw() method

    def update_frame(self, value):
        # self.label.setText(f"Frame: {value}")
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
        self.array3d = array3d
        self.sort_idx = sort_idx
        self.extra_title_strings_list = extra_title_strings_list
        self.num_frames = self.array3d.shape[self.options.slider_axis]
        self.slider.setMaximum(self.num_frames - 1)
        self.spinbox.setMaximum(self.num_frames - 1)
        self.refresh_frame()

    def start(self):
        self.show()


def return_spin_play_slider_widgets(
    num_frames: int, start_index: Optional[int] = 0, parent: Optional[QWidget] = None
) -> tuple[QSlider, QSpinBox, QPushButton, QTimer, QLayout]:
    # Create a slider
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(num_frames - 1)
    slider.setValue(start_index)
    # slider.valueChanged.connect(update_frame)

    # SpinBox (editable + arrows)
    spinbox = QSpinBox()
    spinbox.setMinimum(0)
    spinbox.setMaximum(num_frames - 1)
    spinbox.setValue(start_index)
    slider.valueChanged.connect(spinbox.setValue)
    spinbox.valueChanged.connect(slider.setValue)
    # spinbox.valueChanged.connect(update_frame)
    spinbox.setStyleSheet("""
    QSpinBox {
        font-size: 14px;
        padding: 3px 6px;                /* Inner spacing (top/bottom, left/right) */
        min-width: 60px;                  /* Minimum width */
        min-height: 20px;                 /* Minimum height */
        text-align: center;              /* Text alignment */
    }                           
    """)

    # Play button
    play_button = QPushButton("Play")
    # play_button.clicked.connect(toggle_play)
    play_button.setStyleSheet("""
    QPushButton {
        font-size: 14px;
        padding: 3px 6px;                /* Inner spacing (top/bottom, left/right) */
        min-width: 60px;                  /* Minimum width */
        min-height: 20px;                 /* Minimum height */
        text-align: center;              /* Text alignment */
    }
    """)
    # Package play button, and spin box
    spin_play_layout = QHBoxLayout()
    spin_play_layout.addWidget(play_button)
    spin_play_layout.addWidget(spinbox)
    spin_play_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
    index_selection_layout = QVBoxLayout()
    index_selection_layout.addWidget(slider)
    index_selection_layout.addLayout(spin_play_layout)

    # Timer for playback
    timer = QTimer(parent)
    timer.setInterval(100)  # milliseconds per frame
    # timer.timeout.connect(next_frame)

    return slider, spinbox, play_button, timer, index_selection_layout
