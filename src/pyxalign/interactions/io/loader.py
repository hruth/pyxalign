import matplotlib
import sys
from dataclasses import fields, is_dataclass
from enum import Enum, StrEnum, auto
from typing import Optional, Union, TypeVar, get_origin, get_args, Any
import cupy as cp
import numpy as np
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
from PyQt5.QtCore import Qt
from pyxalign.api.options_utils import get_all_attribute_names

from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.io.loaders.lamni.options import (
    LYNXLoadOptions,
    Beamline2IDELoadOptions,
    BaseLoadOptions,
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


class ExperimentType(StrEnum):
    LYNX = auto()
    BEAMLINE_2IDE_PTYCHO = auto()
    BEAMLINE_2IDE_XRF = auto()


class BasicLoadingOptions(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)


class InteractiveLoadingWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.options_editor = None
        self.options = None

        # self.layout
        self.layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        self.add_options_selector_combo_box()

    def add_options_selector_combo_box(self):
        self.select_options_widget = QWidget()
        layout = QVBoxLayout()
        self.select_options_widget.setLayout(layout)
        self.experiment_type_combo = QComboBox()
        layout.addWidget(self.experiment_type_combo)
        experiment_options = {
            "LYNX": ExperimentType.LYNX,
            "2IDE: ptycho": ExperimentType.BEAMLINE_2IDE_PTYCHO,
            # "2IDE: XRF-maps": ExperimentType.BEAMLINE_2IDE_XRF,
        }
        for key, val in experiment_options.items():
            self.experiment_type_combo.addItem(key, val)

        # On changing the index, the displayed options should change between
        # LYNXLoadOptions, Beamline2IDELoadOptions, and the xrf maps options!
        self.experiment_type_combo.currentIndexChanged.connect(self.change_selected_options_editor)

        self.tabs.addTab(self.select_options_widget, "Select Options")

        self.advanced_options_widget = QWidget()
        self.advanced_options_widget.setLayout(QVBoxLayout())
        self.tabs.addTab(self.advanced_options_widget, "Advanced Options")

        # initialize the options
        self.change_selected_options_editor(idx=self.experiment_type_combo.currentIndex())

    def change_selected_options_editor(self, idx: int):
        selected_experiment = self.experiment_type_combo.itemData(idx)
        # Delete existing widget from layout
        if self.options_editor is not None:
            self.select_options_widget.layout().removeWidget(self.options_editor)
            self.options_editor.deleteLater()
            self.advanced_options_widget.layout().removeWidget(self.options_editor)
            self.advanced_options_editor.deleteLater()
        # Initialize the selected option type
        if selected_experiment is ExperimentType.LYNX:
            self.options = LYNXLoadOptions(
                base=BaseLoadOptions(parent_projections_folder=None), dat_file_path=None
            )
        elif selected_experiment is ExperimentType.BEAMLINE_2IDE_PTYCHO:
            self.options = Beamline2IDELoadOptions(
                base=BaseLoadOptions(parent_projections_folder=None), mda_folder=None
            )
        # Update the options editor
        self.options_editor = BasicOptionsEditor(
            self.options,
            file_dialog_fields=file_dialog_fields,
            folder_dialog_fields=folder_dialog_fields,
            skip_fields=advanced_options_list,
        )
        # Add widget to layout
        self.select_options_widget.layout().addWidget(self.options_editor)
        # Update advanced options
        self.advanced_options_editor = BasicOptionsEditor(
            self.options,
            file_dialog_fields=file_dialog_fields,
            folder_dialog_fields=folder_dialog_fields,
            skip_fields=np.setdiff1d(
                get_all_attribute_names(self.options),
                advanced_options_list,
            ),
        )
        self.advanced_options_widget.layout().addWidget(self.advanced_options_editor)


if __name__ == "__main__":
    app = QApplication([])
    load_widget = InteractiveLoadingWidget()
    screen_geometry = app.desktop().availableGeometry(load_widget)
    load_widget.show()
    sys.exit(app.exec_())
