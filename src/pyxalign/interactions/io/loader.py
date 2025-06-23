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
)
from PyQt5.QtCore import Qt

from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.io.loaders.lamni.options import (
    LYNXLoadOptions,
    Beamline2IDELoadOptions,
    BaseLoadOptions,
)
from pyxalign.io.utils import OptionsClass


class ExperimentType(StrEnum):
    LYNX = auto()
    BEAMLINE_2IDE_PTYCHO = auto()
    BEAMLINE_2IDE_XRF = auto()


class InteractiveLoadingWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.options_editor = None

        # self.layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.add_options_selector_combo_box()
        # self.options_editor = BasicOptionsEditor(default_options, file_dialog_fields=["dat_file_path"])
        # layout.addWidget()

    def add_options_selector_combo_box(self):
        self.experiment_type_combo = QComboBox()
        experiment_options = {
            "LYNX": ExperimentType.LYNX,
            "2IDE: ptycho": ExperimentType.BEAMLINE_2IDE_PTYCHO,
            "2IDE: XRF-maps": ExperimentType.BEAMLINE_2IDE_XRF,
        }
        for key, val in experiment_options.items():
            self.experiment_type_combo.addItem(key, val)

        # On changing the index, the displayed options should change between
        # LYNXLoadOptions, Beamline2IDELoadOptions, and the xrf maps options!

        # def on_index_changed(idx):
        # setattr(self.data_obj, self.field_name, combo.itemData(idx))
        self.experiment_type_combo.currentIndexChanged.connect(self.change_selected_options_editor)
        self.layout.addWidget(self.experiment_type_combo)

        # initialize the options
        self.change_selected_options_editor(idx=self.experiment_type_combo.currentIndex())

    def change_selected_options_editor(self, idx: int):
        selected_experiment = self.experiment_type_combo.itemData(idx)
        
        if self.options_editor is not None:
            self.layout.removeWidget(self.options_editor)
            self.options_editor.deleteLater()

        if selected_experiment is ExperimentType.LYNX:
            self.options_editor = BasicOptionsEditor(
                LYNXLoadOptions(dat_file_path=None), file_dialog_fields=["dat_file_path"]
            )
        elif selected_experiment is ExperimentType.BEAMLINE_2IDE_PTYCHO:
            self.options_editor = BasicOptionsEditor(
                Beamline2IDELoadOptions(mda_folder=None), file_dialog_fields=["mda_folder"]
            )
        self.layout.addWidget(self.options_editor)


if __name__ == "__main__":
    app = QApplication([])
    load_widget = InteractiveLoadingWidget()
    screen_geometry = app.desktop().availableGeometry(load_widget)
    load_widget.show()
    sys.exit(app.exec_())
