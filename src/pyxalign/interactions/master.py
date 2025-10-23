"""
Master widget for the pyxalign GUI application.

This module provides the main application window that coordinates data loading,
projection initialization, and alignment workflows. It uses a sidebar navigation
pattern to organize different stages of the alignment process.
"""
import sys
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt

import pyxalign.data_structures.task as t
from pyxalign.interactions.initialize_projections import (
    MainProjectionTab,
)
from pyxalign.interactions.io.loader import MainLoadingWidget
from pyxalign.interactions.pma_runner import PMAMasterWidget
from pyxalign.interactions.sidebar_navigator import SidebarNavigator
from pyxalign.io.loaders.pear.options import BaseLoadOptions, LYNXLoadOptions
from pyxalign.io.utils import OptionsClass


class MasterWidget(QWidget):
    """
    Main application widget that coordinates the pyxalign GUI workflow.

    This widget provides a sidebar navigation interface for managing the complete
    alignment workflow, from data loading through projection initialization to
    alignment execution. It uses a SidebarNavigator to organize different stages
    and manages signal connections between components.

    Parameters
    ----------
    input_options : OptionsClass
        Configuration options for data loading and processing
    parent : Optional[QWidget], default=None
        Parent widget
    """

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
        self.create_pma_page()

        # connect the signals related to loading "raw" input data
        data_loaded_signal = self.loading_widget.select_load_settings_widget.data_loaded_signal
        data_loaded_signal.connect(self.projection_initializer_widget.on_standard_data_loaded)

        # connect signal so that task is received here once it has been created
        task_initialized_signal = self.projection_initializer_widget.object_created_signal
        task_initialized_signal.connect(self.receive_task)
        # task_initialized_signal.connect(self.phase_unwrap_widget.set_task)

        # connect signal indicating unwrapped phase is now available in the task
        phase_unwrapped_signal = self.projection_initializer_widget.phase_unwrapped_signal
        phase_unwrapped_signal.connect(self.update_pma_page)

    def load_standard_data(self):
        # shortcut to loading data directly without using any io
        self.loading_widget.select_load_settings_widget.load_data()

    def receive_task(self, task: t.LaminographyAlignmentTask):
        print("task received")
        self.task = task

    def create_loading_widget_page(self, input_options: OptionsClass):
        self.loading_widget = MainLoadingWidget(input_options)
        self.navigator.addPage(self.loading_widget, "Load Data")

    def create_projection_initializer_page(self):
        self.projection_initializer_widget = MainProjectionTab()
        self.navigator.addPage(self.projection_initializer_widget, "Projections View")

    def create_pma_page(self):
        # the page contents should only be loaded once phase projections are available
        self.pma_widget = PMAMasterWidget()
        self.navigator.addPage(self.pma_widget, "Projection Matching Alignment")

    def update_pma_page(self):
        # PMAMasterWidget can only be created once unwrapped phase is loaded
        print("Creating PMA page")
        self.pma_widget.initialize_page(self.task)


def launch_master_gui(
    load_options: Optional[OptionsClass] = None,
    wait_until_closed: bool = False,
):
    app = QApplication.instance() or QApplication([])
    gui = MasterWidget(input_options=load_options)
    # window.loading_widget.select_load_settings_widget.load_data()
    screen_geometry = app.desktop().availableGeometry(gui)
    gui.setGeometry(
        screen_geometry.x(),
        screen_geometry.y(),
        int(screen_geometry.width() * 0.75),
        int(screen_geometry.height() * 0.9),
    )
    gui.show()
    gui.setAttribute(Qt.WA_DeleteOnClose)
    gui.show()
    if wait_until_closed:
        app.exec_()
    return gui


if __name__ == "__main__":
    # Pre-load with LYNX options
    options = LYNXLoadOptions(
        dat_file_path="/gdata/LYNX/lamni/2025-1/31ide_2025-03-05/dat-files/tomography_scannumbers.txt",
        # selected_sequences=(2,),
        selected_experiment_name="APS-D_3D",
        base=BaseLoadOptions(
            parent_projections_folder="/gdata/LYNX/lamni/2025-1/31ide_2025-03-05/ptychi_recons/APS_D_3D",
            file_pattern="Ndp128_LSQML_c*_m0.5_p15_cp_mm_opr2_ic/recon_Niter3000.h5",
            select_all_by_default=True,
            # scan_start=252,
            # scan_end=270,
        ),
    )

    # options = LYNXLoadOptions(
    #     dat_file_path="/gpfs/dfnt1/ecu/ecu05/2025-1/31ide_2025-03-05/dat-files/tomography_scannumbers.txt",
    #     selected_sequences=(2,),
    #     selected_experiment_name="APS-D_3D",
    #     base=BaseLoadOptions(
    #         parent_projections_folder="/gpfs/dfnt1/ecu/ecu05/2025-1/31ide_2025-03-05/ptychi_recons/APS_D_3D",
    #         file_pattern="Ndp128_LSQML_c*_m0.5_gaussian_p20_mm_opr2_ic_21/recon_Niter3000.h5",
    #         select_all_by_default=True,
    #     ),
    # )

    # Pre-load with 2IDE options
    # import os
    # folder_name = "2ide/2025-1_Lamni-6"
    # inputs_folder = os.path.join(
    #     os.environ["PYXALIGN_CI_TEST_DATA_DIR"], folder_name, "inputs"
    # )
    # base_load_options = BaseLoadOptions(
    #     parent_projections_folder=os.path.join(inputs_folder, "ptychi_recons"),
    #     file_pattern="Ndp64_LSQML_c*_m0.5_gaussian_p10_mm_ic_pc*ul0.1/recon_Niter5000.h5",
    #     select_all_by_default=True,
    #     scan_start=115,
    #     scan_end=264,
    # )
    # options = Beamline2IDELoadOptions(
    #         mda_folder=os.path.join(inputs_folder, "mda"),
    #         base=base_load_options,
    #     )

    gui = launch_master_gui(load_options=options, wait_until_closed=True)

    # app = QApplication(sys.argv)
    # window = MasterWidget(input_options=options)
    # # window.loading_widget.select_load_settings_widget.load_data()
    # screen_geometry = app.desktop().availableGeometry(window)
    # window.setGeometry(
    #     screen_geometry.x(),
    #     screen_geometry.y(),
    #     int(screen_geometry.width() * 0.75),
    #     int(screen_geometry.height() * 0.9),
    # )

    # window.show()
    # sys.exit(app.exec_())
