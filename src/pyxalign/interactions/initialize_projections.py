from pyxalign.interactions.custom import MultipleOfDivisorSpinBox
from pyxalign.io.loaders.base import StandardData
import matplotlib

import sys
import numpy as np
from typing import Optional, Dict
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QSizePolicy,
)
import pyqtgraph as pg

from pyxalign.io.loaders.utils import convert_projection_dict_to_array


class ProjectionObjectInitializerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.standard_data = None
        self.projection_array = None

        # Main layout (horizontal).
        # Set smaller margins and spacing, and align top-left so that
        # extra space goes to the right/bottom, not between widgets.
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # Anchor layout in the top-left
        self.setLayout(layout)

        self.add_dict_to_array_converter()

    def set_standard_data(self, standard_data: StandardData):
        self.standard_data = standard_data
        print("data received")

    def add_dict_to_array_converter(self):
        # Vertical layout for the sub-elements
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        main_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # Horizontal layout for the spinboxes
        spin_boxes_layout = QHBoxLayout()
        spin_boxes_layout.setContentsMargins(0, 0, 0, 0)
        spin_boxes_layout.setSpacing(5)
        spin_boxes_layout.setAlignment(Qt.AlignLeft)

        new_shape_label = QLabel("Set array shape:")
        self.new_shape_x = MultipleOfDivisorSpinBox()
        self.new_shape_y = MultipleOfDivisorSpinBox()

        # Give spin boxes a fixed or non-expanding horizontal policy
        self.new_shape_x.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.new_shape_y.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        spin_boxes_layout.addWidget(new_shape_label, 0, Qt.AlignLeft)
        spin_boxes_layout.addWidget(self.new_shape_x, 0, Qt.AlignLeft)
        spin_boxes_layout.addWidget(self.new_shape_y, 0, Qt.AlignLeft)

        main_layout.addLayout(spin_boxes_layout)

        create_projection_array_button = QPushButton("Create Projection Array")
        create_projection_array_button.clicked.connect(self.on_create_array_button_clicked)
        # Small or fixed size policy for the button if desired
        create_projection_array_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        main_layout.addWidget(create_projection_array_button)

        self.layout().addLayout(main_layout)

    def on_create_array_button_clicked(self):
        new_shape = (self.new_shape_x.value(), self.new_shape_y.value())
        self.projection_array = convert_projection_dict_to_array(
            self.standard_data.projections,
            delete_projection_dict=False,
            pad_with_mode=True,
            new_shape=new_shape,
        )


def main():
    app = QApplication(sys.argv)

    widget = ProjectionObjectInitializerWidget()
    widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
