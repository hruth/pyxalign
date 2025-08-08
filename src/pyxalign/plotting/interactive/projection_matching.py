from typing import Callable, Optional
import traceback

import pyxalign.alignment.projection_matching as pm
from pyxalign.gpu_utils import return_cpu_array
from pyxalign.plotting.interactive.arrays import ProjectionViewer, VolumeViewer
from pyxalign.plotting.interactive.base import MultiThreadedWidget
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QTabWidget,
    QPushButton,
    QLabel,
)
from PyQt5.QtCore import pyqtSignal, QObject, QMutex, QWaitCondition
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
import numpy as np
from matplotlib.figure import Figure
import matplotlib
from pyxalign.plotting.interactive.utils import OptionsDisplayWidget
from pyxalign.timing.timer_utils import timer, InlineTimer
import cupy as cp

# measuring the timing is not a good idea, it gets all
# messed up when doing multithreading
timer_enabled = False

color_list = list(matplotlib.colors.TABLEAU_COLORS.values())


class PMWorkerSignals(QObject):
    initialize_plots = pyqtSignal()
    update_plots = pyqtSignal()
    update_plots_finished = pyqtSignal()


class ProjectionMatchingViewer(MultiThreadedWidget):
    """Widget for showing overview of projection-matching results"""

    def __init__(
        self,
        pma_object: "pm.ProjectionMatchingAligner",
        multi_thread_func: Optional[Callable] = None,
        parent=None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )
        self.pma_object = pma_object
        self.force_stop = False
        self.setWindowTitle("Projection Matching Alignment Display")
        self.resize(1400, 900)

        # connect signals for initializing the plot
        self.signals = PMWorkerSignals()
        self.signals.initialize_plots.connect(self.initialize_plots)
        self.signals.update_plots.connect(self.update_plots)
        self.signals.update_plots_finished.connect(self.wakeup)

        self.mutex = QMutex()
        self.wait_cond = QWaitCondition()

    def wakeup(self):
        # pass
        self.wait_cond.wakeAll()

    def stop_alignment_thread(self):
        # Shut down thread
        self.force_stop = True
        self.disable_stop_button("terminating alignment loop...please be patient")
        print("terminating alignment loop...")

    def initialize_plots(self, add_stop_button: bool = True):
        try:
            self.set_thread_gpu()
            tabs = QTabWidget()

            # projections viewer
            self.projection_viewer = ProjectionViewer(
                self.pma_object.aligned_projections,
                include_options=False,
                include_shifts=False,
            )
            # volume viewer
            self.volume_viewer = VolumeViewer(self.pma_object.aligned_projections.volume.data)

            # metrics viewer
            metrics_widget = QWidget()
            metrics_layout = QGridLayout()
            metrics_widget.setLayout(metrics_layout)
            self.error_viewer = PMErrorPlotWidget(self.pma_object)  #
            self.shift_viewer = PMShiftPlotWidget(self.pma_object)
            self.shift_diff_viewer = PMShiftDiffPlotWidget(self.pma_object)
            self.error_v_iter_viewer = PMErrorVsIteration(self.pma_object)
            self.step_size_v_iter_viewer = MaxStepSizeVsIteration(self.pma_object)  #
            self.velocity_map_viewer = VelocityMapPlot(self.pma_object)
            self.momentum_v_iter_viewer = MomentumVsIterationMap(self.pma_object)
            metrics_layout.addWidget(self.shift_viewer, 0, 0, 1, 2)
            metrics_layout.addWidget(self.shift_diff_viewer, 1, 0, 1, 2)
            metrics_layout.addWidget(self.error_viewer, 1, 2, 1, 2)  #
            metrics_layout.addWidget(self.step_size_v_iter_viewer, 2, 0, 1, 2)  #
            metrics_layout.addWidget(self.velocity_map_viewer, 0, 3)
            metrics_layout.addWidget(self.momentum_v_iter_viewer, 0, 2)
            metrics_layout.addWidget(self.error_v_iter_viewer, 2, 2, 1, 2)
            metrics_layout.setColumnStretch(0, 1)
            metrics_layout.setColumnStretch(1, 1)
            metrics_layout.setColumnStretch(2, 1)
            metrics_layout.setColumnStretch(3, 1)
            metrics_layout.setRowStretch(0, 1)
            metrics_layout.setRowStretch(1, 1)
            metrics_layout.setRowStretch(2, 1)

            # options viewer
            self.options_display = OptionsDisplayWidget(self.pma_object.options)

            # Add each tab to tab widget
            tabs.addTab(metrics_widget, "Metrics")
            tabs.addTab(self.volume_viewer, "3D Reconstruction")
            tabs.addTab(self.projection_viewer, "Aligned Projections")
            tabs.addTab(self.options_display, "Options")

            # Setup layout
            layout = QVBoxLayout()
            self.plots_layout = QHBoxLayout()
            self.setLayout(layout)
            layout.addLayout(self.plots_layout)
            self.last_update_label = QLabel()
            layout.addWidget(self.last_update_label)
            # Create button for stopping execution of PMA loop
            if add_stop_button:
                self.create_stop_button()
                layout.addWidget(self.stop_button)
            self.plots_layout.addWidget(tabs)

        except (Exception, KeyboardInterrupt) as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()

    def create_stop_button(self):
        self.stop_button = QPushButton(text="STOP ALIGNMENT")
        self.stop_button.clicked.connect(self.stop_alignment_thread)
        self.stop_button.setStyleSheet("QPushButton { background-color: red; color: white; }")

    def disable_stop_button(self, new_text: str):
        self.stop_button.setDisabled(True)
        self.stop_button.setStyleSheet("QPushButton { background-color: gray; }")
        self.stop_button.setText(new_text)

    def finish_test(self):
        self.disable_stop_button("alignment finished")

    @timer(enabled=timer_enabled)
    def update_plots(self):
        try:
            self.set_thread_gpu()
            self.volume_viewer.update_arrays(self.pma_object.aligned_projections.volume.data)
            # self.projection_viewer.update_arrays()#self.pma_object.aligned_projections.data)
            itimer = InlineTimer("shift_viewer")
            itimer.start()
            self.shift_viewer.update_plot()
            itimer.end()

            itimer = InlineTimer("error_viewer")
            itimer.start()
            self.error_viewer.update_plot()
            itimer.end()

            itimer = InlineTimer("shift_diff_viewer")
            itimer.start()
            self.shift_diff_viewer.update_plot()
            itimer.end()

            itimer = InlineTimer("error_v_iter_viewer")
            itimer.start()
            self.error_v_iter_viewer.update_plot()
            itimer.end()

            itimer = InlineTimer("step_size_v_iter_viewer")
            itimer.start()
            self.step_size_v_iter_viewer.update_plot()
            itimer.end()

            itimer = InlineTimer("velocity_map_viewer")
            itimer.start()
            self.velocity_map_viewer.update_plot()
            itimer.end()

            self.momentum_v_iter_viewer.update_plot()

            self.last_update_label.setText(f"Last update: iteration {self.pma_object.iteration}")
        except (Exception, KeyboardInterrupt) as ex:
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            traceback.print_exc()
        finally:
            self.signals.update_plots_finished.emit()

    def set_thread_gpu(self):
        if self.pma_object.options.keep_on_gpu:
            cp.cuda.Device(self.pma_object.options.device.gpu.gpu_indices[0]).use()


class PMLinePlotWidget(QWidget):
    "Base class for basic line plots to be used in the ProjectionMatchingViewer"

    def __init__(self, pma_object: "pm.ProjectionMatchingAligner", parent=None):
        super().__init__(parent)
        self.pma_object = pma_object
        self.sort_idx = np.argsort(self.pma_object.aligned_projections.angles)
        # if cp.get_array_module(self.sort_idx) is cp:
            # self.sort_idx = self.sort_idx.get()
        self.sort_idx = return_cpu_array(self.sort_idx)

        self.figure = Figure(layout="compressed")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setFixedHeight(30)
        self.ax = self.figure.add_subplot(111)
        self.initialize_plot()

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

    @property
    def y_data(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def x_data(self) -> np.ndarray:
        raise NotImplementedError

    @timer(enabled=timer_enabled)
    def update_plot(self):
        itimer = InlineTimer("update line data")
        itimer.start()
        for i, line in enumerate(self.lines):
            if self.y_data.ndim == 1:
                new_data = self.y_data
            elif self.y_data.ndim == 2:
                new_data = self.y_data[:, i]
            line.set_data(self.x_data, return_cpu_array(new_data))
        itimer.end()

        itimer = InlineTimer("rescale, relim, draw_idle")
        itimer.start()
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()
        itimer.end()

    def initialize_plot(self):
        self.ax.clear()
        self.lines = self.ax.plot(self.x_data, return_cpu_array(self.y_data))
        self.initialize_plot_formatting()
        self.canvas.draw_idle()

    def initialize_plot_formatting(self):
        pass

    def add_y_axis_in_microns(self):
        pixel_size_um = self.pma_object.aligned_projections.pixel_size * 1e6
        # Add a second y-axis for nanometers
        self.ax_nm = self.ax.twinx()
        self.ax_nm.set_ylabel(r"Shift ($\mu m$)")

        # Sync the y-axis limits: convert px to nm
        def sync_nm_axis():
            y_min, y_max = self.ax.get_ylim()
            self.ax_nm.set_ylim(y_min * pixel_size_um, y_max * pixel_size_um)

        # Initial sync
        sync_nm_axis()

        # Optionally connect to zoom/pan events to keep them in sync
        def on_ylims_changed(event_ax):
            sync_nm_axis()

        self.ax.callbacks.connect("ylim_changed", on_ylims_changed)


class PMErrorPlotWidget(PMLinePlotWidget):
    def __init__(self, pma_object: "pm.ProjectionMatchingAligner", parent=None):
        super().__init__(pma_object=pma_object, parent=parent)

    @property
    def y_data(self) -> np.ndarray:
        return self.pma_object.all_errors[self.pma_object.iteration, self.sort_idx]

    @property
    def x_data(self) -> np.ndarray:
        return self.pma_object.aligned_projections.angles[self.sort_idx]

    def initialize_plot_formatting(self):
        self.ax.grid(linestyle=":")
        self.ax.set_title("PMA Error")
        self.ax.autoscale(enable=True, axis="x", tight=True)
        self.ax.set_xlabel("Angle (deg)")
        self.ax.set_ylabel("Error")
        self.canvas.draw()


class PMErrorVsIteration(PMLinePlotWidget):
    def __init__(self, pma_object: "pm.ProjectionMatchingAligner", parent=None):
        super().__init__(pma_object=pma_object, parent=parent)

    @property
    def y_data(self) -> np.ndarray:
        return self.pma_object.all_errors[: self.pma_object.iteration].mean(axis=1)

    @property
    def x_data(self) -> np.ndarray:
        return np.arange(0, len(self.y_data), dtype=int)

    def initialize_plot_formatting(self):
        self.ax.grid(linestyle=":")
        self.ax.set_title("Error vs Iteration")
        self.ax.autoscale(enable=True, axis="x", tight=True)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Mean Error")
        self.canvas.draw()


class MaxStepSizeVsIteration(PMLinePlotWidget):
    def __init__(self, pma_object: "pm.ProjectionMatchingAligner", parent=None):
        super().__init__(pma_object=pma_object, parent=parent)

    @property
    def y_data(self) -> np.ndarray:
        return self.pma_object.all_max_shift_step_size

    @property
    def x_data(self) -> np.ndarray:
        return np.arange(0, len(self.y_data), dtype=int)

    def initialize_plot_formatting(self):
        self.ax.grid(linestyle=":")
        self.ax.set_title("Max Step Size Update vs Iteration")
        self.ax.autoscale(enable=True, axis="x", tight=True)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Shift Update (px)")
        self.add_y_axis_in_microns()
        self.canvas.draw()


class PMShiftPlotWidget(PMLinePlotWidget):
    def __init__(self, pma_object: "pm.ProjectionMatchingAligner", parent=None):
        super().__init__(pma_object=pma_object, parent=parent)

    @property
    def y_data(self):
        return self.pma_object.total_shift[self.sort_idx]
        # self.pma_object.total_shift[self.sort_idx]

    @property
    def x_data(self):
        return self.pma_object.aligned_projections.angles[self.sort_idx]

    def initialize_plot(self):
        initial_shift_colors = ((0.75, 0.75, 1), (1, 0.75, 0.75))
        for i in range(2):
            initial_shift = (
                return_cpu_array(self.pma_object.initial_shift[self.sort_idx])
                / self.pma_object.scale
            )
            self.ax.plot(
                self.x_data,
                initial_shift,
                color=initial_shift_colors[i],
            )
        self.lines = self.ax.plot(self.x_data, return_cpu_array(self.y_data))
        self.initialize_plot_formatting()
        self.canvas.draw()

    def initialize_plot_formatting(self):
        self.ax.grid(linestyle=":")
        self.ax.set_title("Alignment Shift")
        self.ax.autoscale(enable=True, axis="x", tight=True)
        self.ax.set_xlabel("Angle (deg)")
        self.ax.set_ylabel("Shift (px)")
        # self.ax.legend(["Horizontal", "Vertical"])
        self.add_y_axis_in_microns()


class VelocityMapPlot(PMLinePlotWidget):
    def __init__(self, pma_object: "pm.ProjectionMatchingAligner", parent=None):
        super().__init__(pma_object=pma_object, parent=parent)

    @property
    def y_data(self):
        return self.pma_object.velocity_map[self.sort_idx]

    @property
    def x_data(self):
        return self.pma_object.aligned_projections.angles[self.sort_idx]

    def initialize_plot_formatting(self):
        self.ax.grid(linestyle=":")
        self.ax.set_title("Velocity Map")
        self.ax.autoscale(enable=True, axis="x", tight=True)
        self.ax.set_xlabel("Angle (deg)")
        self.ax.set_ylabel("Update Velocity")
        # self.ax.legend(["Horizontal", "Vertical"])


class MomentumVsIterationMap(PMLinePlotWidget):
    def __init__(self, pma_object: "pm.ProjectionMatchingAligner", parent=None):
        super().__init__(pma_object=pma_object, parent=parent)

    @property
    def y_data(self):
        return self.pma_object.all_momentum_acceleration

    @property
    def x_data(self) -> np.ndarray:
        return np.arange(0, len(self.y_data), dtype=int)

    def initialize_plot_formatting(self):
        self.ax.grid(linestyle=":")
        self.ax.set_title("Momentum vs Iteration")
        self.ax.autoscale(enable=True, axis="x", tight=True)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Update Momentum")
        # self.ax.legend(["Horizontal", "Vertical"])


class PMShiftDiffPlotWidget(PMLinePlotWidget):
    def __init__(self, pma_object: "pm.ProjectionMatchingAligner", parent=None):
        super().__init__(pma_object=pma_object, parent=parent)

    @property
    def y_data(self):
        return (
            return_cpu_array(self.pma_object.total_shift[self.sort_idx])
            - return_cpu_array(self.pma_object.initial_shift[self.sort_idx]) / self.pma_object.scale
        )

    @property
    def x_data(self):
        return self.pma_object.aligned_projections.angles[self.sort_idx]

    def initialize_plot_formatting(self):
        self.ax.grid(linestyle=":")
        self.ax.set_title("total_shift - initial_shift")
        self.ax.autoscale(enable=True, axis="x", tight=True)
        self.ax.set_xlabel("Angle (deg)")
        self.ax.set_ylabel("Shift (px)")
        self.ax.legend(["Horizontal", "Vertical"])
        self.add_y_axis_in_microns()


