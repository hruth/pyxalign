"""Array save dialog for ArrayViewer."""

from typing import Optional
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QWidget,
    QCheckBox,
)
from PyQt5.QtCore import Qt
import numpy as np


class ArraySaveWindow(QDialog):
    """Dialog for saving arrays from ArrayViewer."""

    def __init__(self, array_viewer, parent: Optional[QWidget] = None):
        """
        Initialize the array save dialog.

        Args:
            array_viewer: The ArrayViewer instance to save data from
            parent: Parent widget
        """
        super().__init__(parent)
        self.array_viewer = array_viewer
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Save Array")
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Section 1: Save mode selection
        mode_group = QGroupBox("Save Mode")
        mode_layout = QVBoxLayout()

        self.radio_3d_tiff = QRadioButton("Save 3D array as TIFF")
        self.radio_current_frame = QRadioButton("Save current frame as image")
        self.radio_3d_tiff.setChecked(True)

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_3d_tiff)
        self.button_group.addButton(self.radio_current_frame)
        self.button_group.buttonClicked.connect(self.on_save_mode_changed)

        mode_layout.addWidget(self.radio_3d_tiff)
        mode_layout.addWidget(self.radio_current_frame)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        # Section 1.5: Sort order selection (only shown if sort_idx is present)
        self.sort_checkbox = QCheckBox("Save sorted array")
        self.sort_checkbox.setChecked(True)  # Default to sorted

        # Only show if array_viewer has a sort_idx
        if self.array_viewer.sort_idx is not None:
            main_layout.addWidget(self.sort_checkbox)
        else:
            self.sort_checkbox.hide()

        # Section 2: Format selection (for single frame)
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["TIFF", "PNG", "JPG"])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        self.format_widget = QWidget()
        self.format_widget.setLayout(format_layout)
        self.format_widget.hide()  # Hidden by default
        main_layout.addWidget(self.format_widget)

        # Section 3: File path selection
        path_layout = QHBoxLayout()
        self.path_line_edit = QLineEdit()
        self.path_line_edit.setPlaceholderText("Select save path...")
        path_layout.addWidget(self.path_line_edit)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_file_path)
        path_layout.addWidget(self.browse_button)
        main_layout.addLayout(path_layout)

        # Section 4: Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.save_button = QPushButton("Save Array")
        self.save_button.clicked.connect(self.perform_save)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

    def on_save_mode_changed(self):
        """Show/hide format dropdown and sort checkbox based on selected mode."""
        if self.radio_current_frame.isChecked():
            self.format_widget.show()
            # Hide sort checkbox for single frame saves
            self.sort_checkbox.hide()
        else:
            self.format_widget.hide()
            # Show sort checkbox only if sort_idx exists
            if self.array_viewer.sort_idx is not None:
                self.sort_checkbox.show()
            else:
                self.sort_checkbox.hide()

    def browse_file_path(self):
        """Open file dialog to select save path."""
        # Determine file filter and default name
        if self.radio_3d_tiff.isChecked():
            filter_str = "TIFF Files (*.tif *.tiff)"
            default_name = "array_3d.tif"
        else:
            format_type = self.format_combo.currentText()
            frame_idx = self.array_viewer.slider.value()
            if format_type == "PNG":
                filter_str = "PNG Files (*.png)"
                default_name = f"frame_{frame_idx}.png"
            elif format_type == "JPG":
                filter_str = "JPEG Files (*.jpg *.jpeg)"
                default_name = f"frame_{frame_idx}.jpg"
            else:  # TIFF
                filter_str = "TIFF Files (*.tif *.tiff)"
                default_name = f"frame_{frame_idx}.tif"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Array", default_name, filter_str
        )

        if file_path:
            self.path_line_edit.setText(file_path)

    def perform_save(self):
        """Execute the save operation."""
        file_path = self.path_line_edit.text()

        if not file_path:
            QMessageBox.warning(self, "No Path", "Please select a save path.")
            return

        try:
            from pyxalign.io.save import (
                save_array_as_tiff,
                save_2d_array_as_png,
                save_2d_array_as_jpg,
                save_2d_array_as_tiff,
            )

            if self.radio_3d_tiff.isChecked():
                # Save full 3D array with proper orientation
                array = self.array_viewer.array3d
                # Convert from GPU if needed
                if hasattr(array, "get"):  # CuPy array
                    array = array.get()

                # Reorganize array to match displayed orientation
                slider_axis = self.array_viewer.options.slider_axis

                # Move slider axis to position 0 (stack direction)
                if slider_axis != 0:
                    array = np.moveaxis(array, slider_axis, 0)

                # Apply sorting if checkbox is checked and sort_idx exists
                if self.sort_checkbox.isChecked() and self.array_viewer.sort_idx is not None:
                    # Reorder the array using sort_idx along axis 0 (after moveaxis)
                    array = array[self.array_viewer.sort_idx]

                # Transpose each 2D slice to match displayed orientation
                # This transposes the last two axes (height, width) for all slices
                array = np.transpose(array, (0, 2, 1))

                # Rotate 90 degrees counterclockwise
                array = np.rot90(array, k=1, axes=(1, 2))

                save_array_as_tiff(array, file_path)
            else:
                # Save current frame
                current_frame = self.array_viewer.get_current_frame_data()

                # Rotate 90 degrees counterclockwise
                current_frame = np.rot90(current_frame, k=1)

                format_type = self.format_combo.currentText()

                if format_type == "PNG":
                    save_2d_array_as_png(current_frame, file_path)
                elif format_type == "JPG":
                    save_2d_array_as_jpg(current_frame, file_path)
                else:  # TIFF
                    save_2d_array_as_tiff(current_frame, file_path)

            QMessageBox.information(self, "Success", f"Array saved to {file_path}")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving array: {str(e)}")
