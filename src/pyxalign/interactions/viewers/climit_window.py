"""
Color limit adjustment window for pyqtgraph ImageItem.

This module provides a simple dialog for adjusting the color limits (climit)
of images displayed in pyqtgraph ImageItem widgets.
"""

from typing import Optional, Tuple
import numpy as np
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QDoubleSpinBox,
    QPushButton,
    QLabel,
    QWidget,
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg


class ClimitAdjustmentWindow(QDialog):
    """
    Simple dialog for adjusting color limits of a pyqtgraph ImageItem.

    Provides two spinboxes for setting lower and upper color limits,
    with Apply and Close buttons for on-demand updates.
    """

    def __init__(self, image_item: pg.ImageItem, parent: Optional[QWidget] = None):
        """
        Initialize the color limit adjustment window.

        Args:
            image_item: The pyqtgraph ImageItem to control
            parent: Parent widget
        """
        super().__init__(parent)
        self.image_item = image_item
        self.setup_ui()
        self.load_current_levels()

    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Adjust Color Limits")
        # self.setModal(True)

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Form layout for spinboxes
        form_layout = QFormLayout()

        # Lower limit spinbox
        self.lower_limit_spinbox = QDoubleSpinBox()
        self.lower_limit_spinbox.setRange(-999999.0, 999999.0)
        self.lower_limit_spinbox.setDecimals(6)
        form_layout.addRow("Lower Limit:", self.lower_limit_spinbox)
        self.lower_limit_spinbox.valueChanged.connect(self.apply_climit)

        # Upper limit spinbox
        self.upper_limit_spinbox = QDoubleSpinBox()
        self.upper_limit_spinbox.setRange(-999999.0, 999999.0)
        self.upper_limit_spinbox.setDecimals(6)
        form_layout.addRow("Upper Limit:", self.upper_limit_spinbox)
        self.upper_limit_spinbox.valueChanged.connect(self.apply_climit)

        main_layout.addLayout(form_layout)

        # Button layout
        button_layout = QHBoxLayout()

        main_layout.addLayout(button_layout)

    def load_current_levels(self):
        """Load current color levels from the image item and populate spinboxes."""
        try:
            current_levels = self.image_item.getLevels()
            if current_levels is not None:
                self.lower_limit_spinbox.setValue(current_levels[0])
                self.upper_limit_spinbox.setValue(current_levels[1])
            else:
                # If no levels are set, use default values
                self.lower_limit_spinbox.setValue(0.0)
                self.upper_limit_spinbox.setValue(1.0)
        except Exception:
            # Fallback to default values if there's any issue
            self.lower_limit_spinbox.setValue(0.0)
            self.upper_limit_spinbox.setValue(1.0)

    def apply_climit(self):
        """Apply the color limits from the spinboxes to the image item."""
        lower_limit = self.lower_limit_spinbox.value()
        upper_limit = self.upper_limit_spinbox.value()

        # Validate that lower < upper
        if lower_limit >= upper_limit:
            # Swap values if they're in wrong order
            lower_limit, upper_limit = upper_limit, lower_limit
            self.lower_limit_spinbox.setValue(lower_limit)
            self.upper_limit_spinbox.setValue(upper_limit)

        # Apply the levels to the image item
        try:
            self.image_item.setLevels([lower_limit, upper_limit])
        except Exception as e:
            print(f"Error applying color limits: {e}")

    def get_current_levels(self) -> Tuple[float, float]:
        """
        Get the current levels from the spinboxes.

        Returns:
            Tuple of (lower_limit, upper_limit)
        """
        return (self.lower_limit_spinbox.value(), self.upper_limit_spinbox.value())
