from . import data_structures
from . import io
from .api import options
from .api import enums

from .interactions.master import launch_master_gui
from .interactions.viewers.launchers import launch_array_viewer, launch_volume_viewer, launch_linked_array_viewer


# Create a gui namespace using a simple class
class _GUI:
    """Namespace for GUI-related functions."""

    launch_master_gui = staticmethod(launch_master_gui)
    launch_array_viewer = staticmethod(launch_array_viewer)
    launch_volume_viewer = staticmethod(launch_volume_viewer)
    launch_linked_array_viewer = staticmethod(launch_linked_array_viewer)
    


gui = _GUI()

__all__ = [
    "data_structures",
    "io",
    "options",
    "enums",
    "gui",
]
