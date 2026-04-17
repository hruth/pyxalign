"""
FSC Calculation Window for interactive Fourier Shell Correlation analysis.

This module provides a GUI for calculating and visualizing FSC from projection data.
"""

from typing import Optional
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSizePolicy, QSpacerItem, QMessageBox, QFileDialog
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

        # Button: Save FSC
        self.save_fsc_button = QPushButton("Save FSC")
        self.save_fsc_button.clicked.connect(self.save_fsc)
        self.save_fsc_button.setDisabled(True)  # Disabled until FSC calculated
        controls_layout.addWidget(self.save_fsc_button)

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
            self.crop_3d_selector_window.close()

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

        # Enable save button after successful calculation
        self.save_fsc_button.setEnabled(True)

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

    def save_fsc(self):
        """Save FSC results to an HDF5 file."""
        if self.projections.fsc is None:
            QMessageBox.warning(
                self, "Error", "Please calculate FSC first!"
            )
            return

        # Open file dialog for selecting save location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save FSC",
            "",
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )

        if file_path:
            # Ensure the file has .h5 extension if no extension provided
            if not file_path.endswith(('.h5', '.hdf5')):
                file_path += '.h5'

            try:
                self.projections.fsc.save_fsc(file_path)
                QMessageBox.information(
                    self, "Success", f"FSC saved successfully to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to save FSC:\n{str(e)}"
                )
