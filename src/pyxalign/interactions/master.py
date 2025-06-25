import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Optional, Union, Callable
import cupy as cp
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pyxalign.api.options.alignment import ProjectionMatchingOptions
from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.options.task import AlignmentTaskOptions
from pyxalign.api.options.transform import DownsampleOptions
from pyxalign.interactions.io.loader import MainLoadingWidget
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.pma_runner import PMAMasterWidget
from pyxalign.interactions.sequencer import SequencerWidget
from pyxalign.interactions.sidebar_navigator import SidebarNavigator
import pyxalign.io.load as load
from pyxalign.plotting.interactive.base import MultiThreadedWidget
import pyxalign.data_structures.task as t

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QToolButton,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QComboBox,
    QPushButton,
    QDialogButtonBox,
    QSizePolicy,
    QHBoxLayout,
    QLayout,
    QScrollArea,
    QSpacerItem,
    QTabWidget,
    QGridLayout,
    QTableWidget,
    QTabBar,
    QAbstractItemView,
    QTableWidgetItem,
    QStackedWidget,
    QToolBar,
    QAction,
    QMainWindow,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon


class MasterWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)

        # Create a vertical layout for the entire widget
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        self.navigator = SidebarNavigator()
        main_layout.addWidget(self.navigator)

        self.initialize_loading_widget_page()

    def initialize_loading_widget_page(self):
        self.loading_widget = MainLoadingWidget()
        self.navigator.addPage(self.loading_widget, "Load")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MasterWidget()
    screen_geometry = app.desktop().availableGeometry(window)
    window.setGeometry(
        screen_geometry.x(),
        screen_geometry.y(),
        int(screen_geometry.width() * 0.75),
        int(screen_geometry.height() * 0.9),
    )

    window.show()
    sys.exit(app.exec_())
