"""Combined data-loader + initialization GUI."""

import copy
from typing import Optional, Tuple

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

from pyxalign.api.types import OptionsClass
from pyxalign.autorunner.config import InitializationConfig
from pyxalign.data_structures.projections import ComplexProjections
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.interactions.autorunner.initialization_widget import (
    InitializationConfigWidget,
)
from pyxalign.interactions.io.loader import SelectLoadSettingsWidget
from pyxalign.interactions.sidebar_navigator import SidebarNavigator
from pyxalign.interactions.utils.misc import (
    center_window_on_screen,
    switch_to_matplotlib_qt_backend,
)


class DataLoadAndInitWidget(QWidget):
    """Sidebar-navigated GUI with 'Data Loader' and 'Initialization' pages.

    Loading data on the first page enables the initialization page and switches to it.
    """

    data_loaded = pyqtSignal()
    finished = pyqtSignal()

    def __init__(
        self,
        load_options: Optional[OptionsClass] = None,
        initialization_config: Optional[InitializationConfig] = None,
        show_finish_button: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Data Loader and Initialization")
        self.resize(1600, 900)

        self.data_loader_widget = SelectLoadSettingsWidget(load_options)
        self.init_widget = InitializationConfigWidget(
            standard_data=None,
            initialization_config=initialization_config,
        )

        self.sidebar = SidebarNavigator()
        self.sidebar.addPage(self.data_loader_widget, "Data Loader")
        self.sidebar.addPage(self.init_widget, "Initialization")
        self._init_action = self.sidebar.action_group.actions()[1]
        self._init_action.setEnabled(False)
        self.init_widget.setEnabled(False)
        self._loaded_options: Optional[OptionsClass] = None

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.sidebar)

        self.finish_button: Optional[QPushButton] = None
        if show_finish_button:
            self.finish_button = QPushButton("Finish")
            self.finish_button.setEnabled(False)
            self.finish_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #2E7D32;
                    color: white;
                    padding: 10px;
                    font-size: 14px;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover { background-color: #1B5E20; }
                QPushButton:disabled { background-color: #adb5bd; color: #6c757d; }
                """
            )
            font = QFont()
            font.setBold(True)
            self.finish_button.setFont(font)
            self.finish_button.clicked.connect(self._on_finish_clicked)
            layout.addWidget(self.finish_button)

        self.data_loader_widget.data_loaded_signal.connect(self._on_data_loaded)

    def _on_data_loaded(self):
        if self.data_loader_widget.loaded_data is None:
            return
        self._loaded_options = copy.deepcopy(self.data_loader_widget.options)
        self.init_widget.setStandardData(self.data_loader_widget.loaded_data)
        self.init_widget.setEnabled(True)
        self._init_action.setEnabled(True)
        self.sidebar.setCurrentPage(1)
        if self.finish_button is not None:
            self.finish_button.setEnabled(True)
        self.data_loaded.emit()

    def _on_finish_clicked(self):
        self.finished.emit()
        self.close()

    @property
    def loading_options(self) -> Optional[OptionsClass]:
        """The options snapshot from the most recent successful load, or None if data was never loaded."""
        return self._loaded_options

    @property
    def standard_data(self):
        return self.data_loader_widget.loaded_data

    def get_or_build_complex_projections(self) -> Optional[ComplexProjections]:
        if self.standard_data is None:
            return None
        return self.init_widget.get_or_build_complex_projections()


@switch_to_matplotlib_qt_backend
def launch_data_loader_and_initialization(
    load_options: Optional[OptionsClass] = None,
    initialization_config: Optional[InitializationConfig] = None,
) -> Tuple[Optional[LaminographyAlignmentTask], Optional[OptionsClass]]:
    """Launch the combined data-loader + initialization GUI.

    Returns a ``(LaminographyAlignmentTask, loading_options)`` tuple. The task
    is ``None`` if the window was closed before any data was loaded.
    """
    app = QApplication.instance() or QApplication([])
    gui = DataLoadAndInitWidget(
        load_options=load_options,
        initialization_config=initialization_config,
    )
    gui.finished.connect(app.quit)
    center_window_on_screen(gui, width_fraction=0.85, height_fraction=0.85)
    gui.show()
    app.exec()

    loading_options = gui.loading_options
    complex_projections = gui.get_or_build_complex_projections()
    gui.close()
    if complex_projections is None:
        return None, loading_options
    task = LaminographyAlignmentTask(complex_projections=complex_projections)
    return task, loading_options