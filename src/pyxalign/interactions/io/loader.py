from email.charset import QP
import traceback
import matplotlib
import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Optional, Union, TypeVar, get_origin, get_args, Any
import cupy as cp
import numpy as np
import multiprocessing as mp
import pyxalign.api.options as opts
import time
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
    QFrame,
    QTabWidget,
    QStackedWidget,
    QProgressDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal
from pyxalign.interactions.utils.loading_display_tools import OverlayWidget, loading_bar_wrapper
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
from pyxalign.io.loaders.load_any import load_dataset_from_arbitrary_options
from pyxalign.io.loaders.xrf.api import load_data_from_xrf_format
import sip
from pyxalign.api.options_utils import get_all_attribute_names
from pyxalign.interactions.io.input_data_viewer import StandardDataViewer
from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.enums import ExperimentType
from pyxalign.io.loaders.pear.api import load_data_from_pear_format
from pyxalign.io.loaders.maps import (
    get_experiment_type_enum_from_options,
    get_loader_options_by_enum,
)
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.io.utils import OptionsClass
from pyxalign.interactions.viewers.utils import OptionsDisplayWidget

import pyxalign.io.loaders.pear.options as pear_options
import pyxalign.io.loaders.xrf.options as xrf_options


advanced_options_list = [
    "base.only_include_files_with",
    "base.exclude_files_with",
    "base.selected_ptycho_strings",
    "base.ask_for_backup_files",
    "base.select_all_by_default",
    "is_tile_scan",
    "n_tiles",
    "selected_tile",
    "selected_sequences",
    "selected_experiment_name",
    "_mda_file_pattern",
    "_lamino_angle_pv_string",
    "_angle_pv_string",
    "_channel_names_path",
    "_channel_data_path",
]

file_dialog_fields = ["dat_file_path"]
folder_dialog_fields = ["base.parent_projections_folder", "mda_folder"]
open_panels_list = ["base"]

T = TypeVar("T", bound=Union[StandardData, dict[str, StandardData]])


class SelectLoadSettingsWidget(QWidget):
    data_loaded_signal = pyqtSignal()
    load_start_signal = pyqtSignal()

    def __init__(
        self, input_options: Optional[OptionsClass] = None, parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.options_editor = None
        self.options = input_options
        self.loaded_data = None
        self.resize(600, 800)

        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.setLayout(layout)
        layout.addWidget(self.tabs)
        self.add_options_selector_combo_box()
        self.add_load_data_button()

    def add_options_selector_combo_box(self):
        self.select_options_widget = QWidget()
        layout = QVBoxLayout()
        self.select_options_widget.setLayout(layout)
        self.experiment_type_combo = QComboBox()
        self.experiment_type_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        layout.addWidget(self.experiment_type_combo)
        experiment_options = {
            "LYNX": ExperimentType.LYNX,
            "2IDE: ptycho": ExperimentType.BEAMLINE_2IDE_PTYCHO,
            "2IDD: ptycho": ExperimentType.BEAMLINE_2IDD_PTYCHO,
            "2IDE: XRF-maps": ExperimentType.BEAMLINE_2IDE_XRF,
        }
        for key, val in experiment_options.items():
            self.experiment_type_combo.addItem(key, val)

        # On changing the index, the displayed options should change between
        # pear_options.LYNXLoadOptions, Beamline2IDELoadOptions, and the xrf maps options!
        self.experiment_type_combo.currentIndexChanged.connect(self.change_selected_options_editor)

        self.tabs.addTab(self.select_options_widget, "Select Options")

        # initialize the options
        if self.options is not None:
            self.insert_options_externally(self.options)
        else:
            self.change_selected_options_editor(idx=self.experiment_type_combo.currentIndex())

    def insert_options_externally(self, options: OptionsClass):
        self.experiment_type_combo.blockSignals(True)
        # find the matching class
        experiment_type = get_experiment_type_enum_from_options(options)
        index = np.where(
            [
                experiment_type == self.experiment_type_combo.itemData(i)
                for i in range(self.experiment_type_combo.count())
            ]
        )[0][0]
        self.experiment_type_combo.setCurrentIndex(index)
        self.display_new_options(options)
        self.experiment_type_combo.blockSignals(False)

    def change_selected_options_editor(self, idx: int):
        # Initialize the selected option type
        selected_experiment = self.experiment_type_combo.itemData(idx)
        self.options = get_loader_options_by_enum(selected_experiment)
        # Update the display
        self.display_new_options(self.options)

    def display_new_options(self, options: Optional[OptionsClass] = None):
        # Delete existing widget from layout
        if self.options_editor is not None:
            self.select_options_widget.layout().removeWidget(self.options_editor)
            self.options_editor.deleteLater()

        basic_options_list = list(
            np.setdiff1d(
                get_all_attribute_names(options),
                advanced_options_list,
            )
        )

        # Update the options editor with advanced tab functionality
        self.options_editor = BasicOptionsEditor(
            options,
            file_dialog_fields=file_dialog_fields,
            folder_dialog_fields=folder_dialog_fields,
            open_panels_list=open_panels_list,
            # advanced_options_list=advanced_options_list,
            basic_options_list=basic_options_list,
            enable_advanced_tab=True,
        )
        # Add widget to layout
        self.select_options_widget.layout().addWidget(self.options_editor)
        # add code for preventing the "view selections" widget from opening a second window

    def add_load_data_button(self):
        self.load_data_button = QPushButton("Load Data")
        self.load_data_button.clicked.connect(self.load_data)
        self.load_data_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.load_data_button.setStyleSheet(
            "QPushButton { background-color: green; font-weight: bold; font-size: 11pt; color: white; padding: 2px 6px;}"
        )
        self.layout().addWidget(self.load_data_button)

    def load_data(self):
        self.load_start_signal.emit()
        try:
            self.overlay = OverlayWidget(self, "Loading data...")
            self.overlay.show()
            self.overlay.raise_()
            self.setEnabled(False)
            QApplication.processEvents()
            self.loaded_data = load_dataset_from_arbitrary_options(
                self.options, int(mp.cpu_count() * 0.8)
            )
            self.data_loaded_signal.emit()
            print("Data loading completed!")
        except Exception as ex:
            print(f"An error occurred during data loading: {type(ex).__name__}: {str(ex)}")
        self.overlay.hide()
        self.setEnabled(True)


class MainLoadingWidget(QWidget):
    def __init__(
        self, input_options: Optional[OptionsClass] = None, parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.input_options = input_options
        self.select_load_settings_widget = SelectLoadSettingsWidget(input_options)
        self.select_load_settings_widget.data_loaded_signal.connect(self.on_data_loaded)
        self.select_load_settings_widget.load_start_signal.connect(self.on_load_data_button_pushed)
        # self.resize(800, 800)

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        self.open_file_loader_button = QPushButton("Load Projections")
        self.open_file_loader_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.open_file_loader_button.setStyleSheet("""
            QPushButton {
                font-size: 12pt; 
                padding: 4px 6px;
            }
            """)
        self.open_file_loader_button.clicked.connect(self.open_file_loader_window)

        self.options_display = OptionsDisplayWidget()

        # self.select_load_settings_widget = SelectLoadSettingsWidget(input_options) # need way to save and load options

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.open_file_loader_button)  # , alignment=Qt.AlignHCenter)
        left_layout.addWidget(self.options_display)
        main_layout.addLayout(left_layout)

        self.standard_data_viewer = StandardDataViewer()
        main_layout.addWidget(self.standard_data_viewer)

        self.standard_data_viewer.setDisabled(True)
        self.options_display.setDisabled(True)

    def on_load_data_button_pushed(self):
        # would be better to have a method that resets the view,
        # but no easy way to do this at the moment
        pass

    def on_data_loaded(self):
        # should be modified to work with xrf data as well
        self.standard_data_viewer.setDisabled(False)
        self.options_display.setDisabled(False)

        self.standard_data_viewer.setStandardData(self.loaded_data)
        self.options_display.update_options(self.select_load_settings_widget.options)
        self.options_display.update_display()

    def open_file_loader_window(self):
        self.select_load_settings_widget.show()


@switch_to_matplotlib_qt_backend
def launch_data_loader(load_options: Optional[OptionsClass] = None) -> T:
    """Launch a GUI for filling out load options and loading data.

    Args:
        load_options (OptionsClass): Load options to be filled out upon
            opening the GUI.

    Returns:
        The loaded data

    Example:
        Launch the data loader
        GUI::

            loaded_data = pyxalign.gui.launch_data_loader()
    """
    app = QApplication.instance() or QApplication([])
    gui = SelectLoadSettingsWidget(load_options)

    # Define a slot to handle the signal containing the
    # loaded data
    result = {}

    def on_data_loaded():
        result["data"] = gui.loaded_data
        app.quit()

    gui.data_loaded_signal.connect(on_data_loaded)

    gui.show()
    app.exec()
    gui.close()
    if result != {}:
        # Return the result after the app closes
        return result["data"]


if __name__ == "__main__":
    options = pear_options.LYNXLoadOptions(
        dat_file_path="/gpfs/dfnt1/ecu/ecu05/2025-1/31ide_2025-03-05/dat-files/tomography_scannumbers.txt",
        selected_sequences=(2,),
        selected_experiment_name="APS-D_3D",
        base=pear_options.BaseLoadOptions(
            parent_projections_folder="/gpfs/dfnt1/ecu/ecu05/2025-1/31ide_2025-03-05/ptychi_recons/APS_D_3D",
            file_pattern="Ndp128_LSQML_c*_m0.5_gaussian_p20_mm_opr2_ic_21/recon_Niter3000.h5",
            select_all_by_default=True,
        ),
    )

    app = QApplication([])
    # load_widget = SelectLoadSettingsWidget(options)
    load_widget = MainLoadingWidget(options)
    screen_geometry = app.desktop().availableGeometry(load_widget)
    load_widget.show()
    sys.exit(app.exec_())

    # standard_data = load_data_from_pear_format(
    #     options=options,
    #     n_processes=int(mp.cpu_count() * 0.8),
    # )
