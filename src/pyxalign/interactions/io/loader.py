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
)
from PyQt5.QtCore import Qt, pyqtSignal
from pyxalign.api.options_utils import get_all_attribute_names
from pyxalign.interactions.io.input_data_viewer import StandardDataViewer
from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.enums import ExperimentType
from pyxalign.io.loaders.pear.api import load_data_from_pear_format
from pyxalign.io.loaders.maps import (
    get_experiment_type_enum_from_options,
    get_loader_options_by_enum,
)
from pyxalign.io.loaders.xrf.options import Beamline2IDEXRFLoadOptions
import sip

from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.io.loaders.pear.options import (
    LYNXLoadOptions,
    Beamline2IDELoadOptions,
    BaseLoadOptions,
    PEARLoadOptions,
)
from pyxalign.io.utils import OptionsClass
from pyxalign.plotting.interactive.utils import OptionsDisplayWidget

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
]

file_dialog_fields = ["dat_file_path"]
folder_dialog_fields = ["base.parent_projections_folder", "mda_folder"]
open_panels_list = ["base"]


class SelectLoadSettingsWidget(QWidget):
    data_loaded_signal = pyqtSignal(StandardData)

    def __init__(
        self, input_options: Optional[OptionsClass] = None, parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.options_editor = None
        self.options = input_options
        self.resize(600, 800)

        # self.layout
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.setLayout(layout)
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
        # LYNXLoadOptions, Beamline2IDELoadOptions, and the xrf maps options!
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
        try:
            if isinstance(self.options, PEARLoadOptions):
                standard_data = load_data_from_pear_format(
                    options=self.options,
                    n_processes=int(mp.cpu_count() * 0.8),
                )
            elif isinstance(self.options, Beamline2IDEXRFLoadOptions):
                pass
            print("Data loading completed!")
            self.data_loaded_signal.emit(standard_data)
        except Exception as ex:
            print(f"An error occurred during data loading: {type(ex).__name__}: {str(ex)}")


class MainLoadingWidget(QWidget):
    def __init__(
        self, input_options: Optional[OptionsClass] = None, parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.input_options = input_options
        self.select_load_settings_widget = SelectLoadSettingsWidget(input_options)
        self.select_load_settings_widget.data_loaded_signal.connect(self.on_data_loaded)
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
        left_layout.addWidget(self.open_file_loader_button)#, alignment=Qt.AlignHCenter)
        left_layout.addWidget(self.options_display)
        main_layout.addLayout(left_layout)

        self.standard_data_viewer = StandardDataViewer()
        main_layout.addWidget(self.standard_data_viewer)

        self.standard_data_viewer.setDisabled(True)
        self.options_display.setDisabled(True)

    def on_data_loaded(self, input_data: StandardData):
        self.standard_data_viewer.setDisabled(False)
        self.options_display.setDisabled(False)
    
        self.standard_data_viewer.setStandardData(input_data)
        self.options_display.update_options(self.select_load_settings_widget.options)
        self.options_display.update_display()

    def open_file_loader_window(self):
        self.select_load_settings_widget.show()


# class MainLoadingWidget(QWidget):
#     def __init__(self, parent: Optional[QWidget] = None):
#         super().__init__(parent)
#         self.select_load_settings_widget = None
#         self.resize(800, 800)

#         layout = QVBoxLayout()
#         self.setLayout(layout)

#         self.open_file_loader_button = QPushButton("Load Projections")
#         self.open_file_loader_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.open_file_loader_button.clicked.connect(self.open_file_loader_window)
#         layout.addWidget(self.open_file_loader_button)

#     def open_file_loader_window(self):
#         if self.select_load_settings_widget is None or sip.isdeleted(
#             self.select_load_settings_widget
#         ):
#             self.select_load_settings_widget = SelectLoadSettingsWidget()
#         self.select_load_settings_widget.setAttribute(Qt.WA_DeleteOnClose)
#         self.select_load_settings_widget.show()


if __name__ == "__main__":
    options = LYNXLoadOptions(
        dat_file_path="/gpfs/dfnt1/ecu/ecu05/2025-1/31ide_2025-03-05/dat-files/tomography_scannumbers.txt",
        selected_sequences=(2,),
        selected_experiment_name="APS-D_3D",
        base=BaseLoadOptions(
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
