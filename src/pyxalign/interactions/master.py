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
from pyxalign.interactions.initialize_projections import (
    CreateProjectionArrayWidget,
    ProjectionInitializerWidget,
)
from pyxalign.interactions.io.loader import MainLoadingWidget
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.pma_runner import PMAMasterWidget
from pyxalign.interactions.sequencer import SequencerWidget
from pyxalign.interactions.sidebar_navigator import SidebarNavigator
import pyxalign.io.load as load
from pyxalign.io.loaders.lamni.options import BaseLoadOptions, LYNXLoadOptions
from pyxalign.io.utils import OptionsClass
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
    def __init__(self, input_options: OptionsClass, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)
        self.task = None

        # Create a vertical layout for the entire widget
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        self.navigator = SidebarNavigator()
        main_layout.addWidget(self.navigator)

        self.create_loading_widget_page(input_options)
        self.create_projection_initializer_page()

        # connect the signals related to loading "raw" input data
        data_loaded_signal = self.loading_widget.select_load_settings_widget.data_loaded_signal
        data_loaded_signal.connect(self.projection_initializer_widget.on_standard_data_loaded)

        # connect signal so that task is received here once it has been created
        task_initialized_signal = (
            self.projection_initializer_widget.object_created_signal
        )
        task_initialized_signal.connect(self.receive_task)
    
    def receive_task(self, task: t.LaminographyAlignmentTask):
        self.task = task

    def create_loading_widget_page(self, input_options: OptionsClass):
        self.loading_widget = MainLoadingWidget(input_options)
        self.navigator.addPage(self.loading_widget, "Load Data")

    def create_projection_initializer_page(self):
        self.projection_initializer_widget = ProjectionInitializerWidget()
        self.navigator.addPage(self.projection_initializer_widget, "Projections View")


if __name__ == "__main__":
    # options = LYNXLoadOptions(
    #     dat_file_path="/gpfs/dfnt1/ecu/ecu05/2025-1/31ide_2025-03-05/dat-files/tomography_scannumbers.txt",
    #     selected_sequences=(2,),
    #     selected_experiment_name="APS-D_3D",
    #     base=BaseLoadOptions(
    #         parent_projections_folder="/gpfs/dfnt1/ecu/ecu05/2025-1/31ide_2025-03-05/ptychi_recons/APS_D_3D",
    #         file_pattern="Ndp128_LSQML_c*_m0.5_gaussian_p20_mm_opr2_ic_21/recon_Niter3000.h5",
    #         select_all_by_default=True,
    #         scan_start=252,
    #         scan_end=270,
    #     ),
    # )
    options = LYNXLoadOptions(
        dat_file_path="/gdata/LYNX/lamni/2025-1/31ide_2025-03-05/dat-files/tomography_scannumbers.txt",
        selected_sequences=(2,),
        selected_experiment_name="APS-D_3D",
        base=BaseLoadOptions(
            parent_projections_folder="/gdata/LYNX/lamni/2025-1/31ide_2025-03-05/ptychi_recons/APS_D_3D",
            file_pattern="Ndp128_LSQML_c*_m0.5_p15_cp_mm_opr2_ic/recon_Niter3000.h5",
            select_all_by_default=True,
            scan_start=252,
            scan_end=270,
        ),
    )
    

    app = QApplication(sys.argv)
    window = MasterWidget(input_options=options)
    screen_geometry = app.desktop().availableGeometry(window)
    window.setGeometry(
        screen_geometry.x(),
        screen_geometry.y(),
        int(screen_geometry.width() * 0.75),
        int(screen_geometry.height() * 0.9),
    )

    window.show()
    sys.exit(app.exec_())
