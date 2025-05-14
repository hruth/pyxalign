from typing import Optional
import numpy as np
from pyxalign.api.options.plotting import ProjectionViewerOptions
from pyxalign.plotting.interactive.arrays import ProjectionViewer, VolumeViewer
import pyxalign.data_structures.projections as p
from PyQt5.QtWidgets import QApplication


def launch_volume_viewer(array_3d: np.ndarray) -> VolumeViewer:
    app = QApplication.instance() or QApplication([])
    gui = VolumeViewer(volume=array_3d)
    gui.show()
    return gui


def launch_projection_viewer(
    projections: "p.Projections", options: Optional[ProjectionViewerOptions] = None
) -> ProjectionViewer:
    app = QApplication.instance() or QApplication([])
    gui = ProjectionViewer(projections, options)
    gui.show()
    return gui
