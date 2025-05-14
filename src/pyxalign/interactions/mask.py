import sys
import numpy as np
import matplotlib
# Use the Qt5Agg backend for embedding in PyQt5
# matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QDoubleSpinBox,
    QPushButton,
    QLabel,
)
from PyQt5.QtCore import Qt, QTimer


class PlotCanvas(FigureCanvas):
    """
    A custom FigureCanvas class to embed a matplotlib plot in a PyQt5 widget.
    """

    def __init__(self, parent=None):
        # Create a figure and a set of subplots (1 row, 3 columns)
        self.fig, self.axes = plt.subplots(1, 3, figsize=(9, 3))
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()

    def update_mask_plot(self, masks, projections, idx, threshold):
        """
        Update the figure with the mask, the masked projection,
        and (1 - mask) * projection, based on the given index and threshold.
        """
        # Clear all subplots before plotting
        for ax in self.axes:
            ax.clear()

        # Create a binary mask clip based on the threshold
        clipped_masks = (masks[idx] * 1).copy()
        clip_idx = clipped_masks > threshold
        clipped_masks[:] = 0
        clipped_masks[clip_idx] = 1

        # Plot data
        ax0, ax1, ax2 = self.axes

        # 1) Mask
        ax0.imshow(clipped_masks, cmap="gray", vmin=0, vmax=1)
        ax0.set_title("Mask")

        # 2) Mask × Projection (angle)
        if np.isreal(projections[0, 0, 0]):

            def process_func(x):
                return x
        else:
            process_func = np.angle
        ax1.imshow(clipped_masks * process_func(projections[idx]), cmap="gray")
        ax1.set_title(r"Mask $\times$ Projection")

        # 3) (1 - Mask) × Projection (angle)
        ax2.imshow((1 - clipped_masks) * process_func(projections[idx]), cmap="gray")
        ax2.set_title(r"(1 - Mask) $\times$ Projection")

        self.fig.tight_layout()
        self.draw()


class ThresholdSelector(QWidget):
    """
    A PyQt5 widget that encapsulates:
    - A slider for frame index
    - A spin box for threshold
    - A 'Play' button to iterate through frames
    - A 'Stop' button that finalizes the threshold
    - An embedded matplotlib plot (PlotCanvas)
    """

    def __init__(self, masks: np.ndarray, projections: np.ndarray, init_thresh: float = 0.0):
        super().__init__()

        self.masks = masks
        self.projections = projections
        self.num_frames = len(masks)
        self.is_final_value = False
        self.threshold = init_thresh

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Plot canvas
        self.plot_canvas = PlotCanvas(self)
        main_layout.addWidget(self.plot_canvas)

        # --- Controls Layout ---
        controls_layout = QHBoxLayout()

        # 1) Frame index slider
        self.index_slider = QSlider(Qt.Horizontal)
        self.index_slider.setRange(0, self.num_frames - 1)
        self.index_slider.setValue(0)
        self.index_slider.valueChanged.connect(self.on_slider_changed)
        controls_layout.addWidget(QLabel("Index"))
        controls_layout.addWidget(self.index_slider)

        # 2) Threshold spin box
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 1e9)
        self.threshold_spin.setValue(init_thresh)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.valueChanged.connect(self.on_threshold_changed)
        controls_layout.addWidget(QLabel("Threshold"))
        controls_layout.addWidget(self.threshold_spin)

        # 3) Play button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.start_playing)
        controls_layout.addWidget(self.play_button)

        # 4) Stop button
        self.stop_button = QPushButton("Select and Finish")
        self.stop_button.setStyleSheet("background-color: blue; color: white;")
        self.stop_button.clicked.connect(self.stop_interaction)
        controls_layout.addWidget(self.stop_button)

        main_layout.addLayout(controls_layout)

        # Timer for auto-play
        self.timer = QTimer()
        self.timer.setInterval(500)  # milliseconds per frame
        self.timer.timeout.connect(self.advance_frame)

        # Initial Plot
        self.update_plot()

        # Widget title/size
        self.setWindowTitle("Threshold Selector (PyQt5)")
        self.resize(900, 600)

    def start_playing(self):
        """
        Start the timer to auto-increment the slider (similar to 'Play').
        """
        if not self.timer.isActive():
            self.timer.start()
            self.play_button.setText("Pause")
        else:
            # Pause if it's already running
            self.timer.stop()
            self.play_button.setText("Play")

    def advance_frame(self):
        """
        Auto-increment the slider's current value if possible.
        """
        current_value = self.index_slider.value()
        if current_value < self.num_frames - 1:
            self.index_slider.setValue(current_value + 1)
        else:
            # If we're at the last frame, wrap around or stop
            self.index_slider.setValue(0)

    def on_slider_changed(self):
        """
        Called whenever the frame slider value changes.
        """
        self.update_plot()

    def on_threshold_changed(self):
        """
        Called whenever the threshold spin box value changes.
        """
        self.threshold = self.threshold_spin.value()
        self.update_plot()

    def update_plot(self):
        """
        Update the embedded plot given the current index and threshold.
        """
        idx = self.index_slider.value()
        self.threshold = self.threshold_spin.value()
        self.plot_canvas.update_mask_plot(self.masks, self.projections, idx, self.threshold)

    def stop_interaction(self):
        """
        Disable all interactivity and set the final threshold.
        """
        self.is_final_value = True
        self.timer.stop()
        plt.close(self.plot_canvas.fig)

        # Disable widgets
        self.index_slider.setEnabled(False)
        self.threshold_spin.setEnabled(False)
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        

        # Here you could do additional cleanup if desired
        print(f"Final threshold value: {self.threshold}")

        self.close()


def illum_map_threshold_plotter(
    masks: np.ndarray, projections: np.ndarray, init_thresh: float = 0.0
) -> ThresholdSelector:
    """
    Creates and returns a ThresholdSelector PyQt5 widget.
    """
    selector_widget = ThresholdSelector(masks, projections, init_thresh)
    return selector_widget


# def main():
#     """
#     Demo usage, showing how one might run the widget in a standalone PyQt application.
#     """
#     app = QApplication(sys.argv)

#     # Generate dummy data for demonstration
#     # masks: a 3D array of shape (num_frames, height, width)
#     # projections: a 3D array of shape (num_frames, height, width)
#     num_frames = 20
#     height, width = 50, 50
#     dummy_masks = np.random.rand(num_frames, height, width)
#     dummy_projections = np.random.rand(num_frames, height, width) * np.exp(1j * np.random.rand(num_frames, height, width))

#     # Create the selector widget
#     selector = illum_map_threshold_plotter(dummy_masks, dummy_projections, init_thresh=0.5)
#     selector.show()

#     sys.exit(app.exec_())


# if __name__ == "__main__":
#     main()
