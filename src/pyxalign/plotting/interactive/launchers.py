from typing import Optional, Sequence, Callable
import numpy as np
from pyxalign.api.options.plotting import ProjectionViewerOptions
from pyxalign.plotting.interactive.arrays import ProjectionViewer, VolumeViewer
from pyxalign.plotting.interactive.base import ArrayViewer
import pyxalign.data_structures.projections as p
from pyxalign.api.options.plotting import ArrayViewerOptions

# import pyxalign.data_structures.xrf_task as x
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# from pyxalign.plotting.interactive.xrf import XRFProjectionsViewer, XRFVolumeViewer


def launch_volume_viewer(array_3d: np.ndarray) -> VolumeViewer:
    app = QApplication.instance() or QApplication([])
    gui = VolumeViewer(volume=array_3d)
    gui.show()
    return gui


def launch_projection_viewer(
    projections: "p.Projections",
    options: Optional[ProjectionViewerOptions] = None,
    enable_dropping: bool = True,
    wait_until_closed: bool = False,
) -> ProjectionViewer:
    app = QApplication.instance() or QApplication([])
    gui = ProjectionViewer(projections, options, enable_dropping=enable_dropping)
    gui.show()
    if wait_until_closed:
        app.exec_()
    return gui

def launch_array_viewer(
    array3d: np.ndarray,
    options: Optional[ArrayViewerOptions] = None,
    sort_idx: Optional[Sequence] = None,
    extra_title_strings_list: Optional[list[str]] = None,
    process_func: Optional[Callable] = None,
    wait_until_closed: bool = False,
):
    app = QApplication.instance() or QApplication([])
    gui = ArrayViewer(
        array3d,
        options,
        sort_idx,
        extra_title_strings_list=extra_title_strings_list,
        process_func=process_func,
    )
    gui.setAttribute(Qt.WA_DeleteOnClose)
    gui.show()
    if wait_until_closed:
        app.exec_()
    return gui


# def launch_xrf_projections_viewer(xrf_task: "x.xrf_task") -> XRFProjectionsViewer:
#     app = QApplication.instance() or QApplication([])
#     gui = XRFProjectionsViewer(xrf_task)
#     gui.show()
#     return gui


# def launch_xrf_volume_viewer(xrf_task: "x.xrf_task") -> XRFVolumeViewer:
#     app = QApplication.instance() or QApplication([])
#     gui = XRFVolumeViewer(xrf_task)
#     gui.show()
#     return gui
