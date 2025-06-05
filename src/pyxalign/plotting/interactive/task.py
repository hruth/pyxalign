from PyQt5.QtWidgets import QWidget, QTabWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from pyxalign.api.options.plotting import ProjectionViewerOptions
import pyxalign.data_structures.task as t
from pyxalign.plotting.interactive.arrays import ProjectionViewer, VolumeViewer
from pyxalign.plotting.interactive.projection_matching import ProjectionMatchingViewer


class TaskViewer(QWidget):
    def __init__(self, task: "t.LaminographyAlignmentTask"):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Task Overview")
        self.resize(1400, 900)

        tabs = QTabWidget()
        if task.complex_projections is not None:
            # Complex projections tab
            tabs.addTab(
                ProjectionViewer(task.complex_projections, ProjectionViewerOptions()),
                "Complex Projections",
            )

        if task.phase_projections is not None:
            # Phase projections tab
            tabs.addTab(
                ProjectionViewer(task.phase_projections, ProjectionViewerOptions()),
                "Unwrapped Projections",
            )
            # 3D volume tab
            if (
                task.phase_projections.volume is not None
                and task.phase_projections.volume.data is not None
            ):
                tabs.addTab(
                    VolumeViewer(task.phase_projections.volume.data),
                    "3D Reconstruction",
                )

        if task.pma_object is not None:
            pma_gui = ProjectionMatchingViewer(task.pma_object)
            pma_gui.initialize_plots(add_stop_button=False)
            pma_gui.update_plots()
            tabs.addTab(pma_gui, "Projection Matching")

        layout = QVBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)

    def start(self):
        self.show()
