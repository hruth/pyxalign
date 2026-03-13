"""
FSC Calculation Window for interactive Fourier Shell Correlation analysis.

This module provides a GUI for calculating and visualizing FSC from projection data.
"""

from typing import Optional
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSizePolicy, QSpacerItem, QMessageBox
)
from PyQt5.QtCore import Qt

from pyxalign.data_structures.projections import Projections
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.crop_3d_selector import GetCrop3DOptionsFromSelector
from pyxalign.interactions.utils.loading_display_tools import loading_bar_wrapper


class FSCCalculationWindow(QWidget):
    """
    Interactive window for Fourier Shell Correlation calculation and visualization.

    This window provides controls for:
    - Calculating volumes for FSC analysis
    - Configuring FSC options (excluding 3D crop, which has dedicated selector)
    - Selecting 3D crop region interactively
    - Calculating FSC between volumes
    - Visualizing FSC results with threshold curves

    Parameters
    ----------
    projections : Projections
        The projections object containing data and methods for FSC calculation
    parent : Optional[QWidget]
        Optional parent widget
    """

    def __init__(self, projections: Projections, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.projections = projections
        self.volumes_for_fsc = None
        self.crop_3d_selector_window = None

        self.setup_ui()
        self.setWindowTitle("FSC Calculation")
        self.resize(1200, 700)

    def setup_ui(self):
        """Setup the user interface layout."""
        # Main layout (horizontal - left controls, right plot)
        main_layout = QHBoxLayout(self)

        # Left panel (controls)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Button: Calculate volumes
        self.calc_volumes_button = QPushButton("Calculate Volumes for FSC")
        self.calc_volumes_button.clicked.connect(self.calculate_volumes)
        controls_layout.addWidget(self.calc_volumes_button)

        # BasicOptionsEditor for FSC options
        controls_layout.addWidget(QLabel("FSC Options:"))
        self.options_editor = BasicOptionsEditor(
            data=self.projections.options.fsc,
            skip_fields=["crop_3d"],
        )
        controls_layout.addWidget(self.options_editor)

        # Button: Select 3D Crop
        self.select_crop_button = QPushButton("Select 3D Crop")
        self.select_crop_button.clicked.connect(self.open_crop_3d_selector)
        self.select_crop_button.setDisabled(True)  # Disabled until volumes calculated
        controls_layout.addWidget(self.select_crop_button)

        # Button: Calculate FSC
        self.calc_fsc_button = QPushButton("Calculate Fourier Shell Correlation")
        self.calc_fsc_button.clicked.connect(self.calculate_fsc)
        self.calc_fsc_button.setDisabled(True)  # Disabled until volumes calculated
        controls_layout.addWidget(self.calc_fsc_button)

        # Spacer
        controls_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # Right panel (matplotlib plot)
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Fourier Shell Correlation")
        self.ax.set_xlabel(r"spatial frequency $\mu m ^{-1}$")
        self.ax.set_ylabel("FSC")
        self.ax.grid(ls=":")

        # Add to main layout
        main_layout.addWidget(controls_widget, stretch=1)
        main_layout.addWidget(self.canvas, stretch=3)

    def calculate_volumes(self):
        """Calculate volumes for FSC analysis."""
        def _calculate():
            self.volumes_for_fsc = self.projections.get_volumes_for_fourier_shell_correlation()

        calc_wrapped = loading_bar_wrapper("Calculating volumes for FSC...")(_calculate)
        calc_wrapped()

        # Enable crop and FSC buttons after successful calculation
        self.select_crop_button.setEnabled(True)
        self.calc_fsc_button.setEnabled(True)

        QMessageBox.information(
            self, "Success", "Volumes calculated successfully!"
        )

    def open_crop_3d_selector(self):
        """Open the 3D crop selector window."""
        if self.volumes_for_fsc is None:
            QMessageBox.warning(
                self, "Error", "Please calculate volumes first!"
            )
            return

        try:
            if self.crop_3d_selector_window is not None:
                self.crop_3d_selector_window.isVisible()
        except RuntimeError:
            self.crop_3d_selector_window = None

        if self.crop_3d_selector_window is None:
            self.crop_3d_selector_window = GetCrop3DOptionsFromSelector(
                array3d=self.volumes_for_fsc[0],
                crop_options=self.projections.options.fsc.crop_3d,
            )
            self.crop_3d_selector_window.crop_3d_selected.connect(
                self.on_crop_selected
            )

        self.crop_3d_selector_window.show()

    def on_crop_selected(self):
        """Handle crop selection completion."""
        if self.crop_3d_selector_window is not None:
            self.projections.options.fsc.crop_3d = (
                self.crop_3d_selector_window.options
            )

    def calculate_fsc(self):
        """Calculate Fourier Shell Correlation."""
        if self.volumes_for_fsc is None:
            QMessageBox.warning(
                self, "Error", "Please calculate volumes first!"
            )
            return

        def _calculate():
            self.projections.get_fourier_shell_correlation(
                volumes=self.volumes_for_fsc
            )

        calc_wrapped = loading_bar_wrapper("Calculating FSC...")(_calculate)
        calc_wrapped()

        # Plot the results
        self.plot_fsc()

    def plot_fsc(self):
        """Plot the FSC results."""
        from pyxalign.fsc import one_half_bit_threshold, get_resolution_crossing

        self.ax.clear()

        # Get FSC data
        fsc_obj = self.projections.fsc
        plot_f = fsc_obj.f * 1e-6  # Convert to MHz

        # Plot FSC curve
        ln, = self.ax.plot(plot_f, fsc_obj.fsc, label=None)

        # Add threshold curve (using half-bit by default)
        threshold_curve = one_half_bit_threshold(fsc_obj.n_shell, 1, 1)
        self.ax.plot(plot_f, threshold_curve, 'k:', label='1/2-bit threshold')

        # Plot resolution crossing
        f_crossing, resolution, crossing_exists = get_resolution_crossing(
            fsc_obj.fsc, threshold_curve, fsc_obj.f
        )
        if crossing_exists:
            self.ax.axvline(f_crossing * 1e-6, color=ln.get_color(), ls='--')
            print(f"Resolution crossing: {resolution * 1e9:0.2f} nm")
        else:
            print("No resolution crossing")

        # Set labels and styling
        self.ax.set_xlabel(r"spatial frequency $\mu m ^{-1}$")
        self.ax.set_ylabel("FSC")
        self.ax.set_title("Fourier Shell Correlation")
        self.ax.grid(ls=":")
        self.ax.autoscale(True, 'x', True)
        self.ax.set_ylim([0, 1.01])
        self.ax.legend()

        # Redraw the canvas
        self.canvas.draw()
