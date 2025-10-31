from IPython import get_ipython

# from .master import launch_master_gui

# __all__ = ["interactive"]


ipython = get_ipython()
if ipython:
    # ipython.magic("gui qt")
    ipython.run_line_magic("gui", "qt")
