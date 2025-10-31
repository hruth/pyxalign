from IPython import get_ipython
ipython = get_ipython()
if ipython:
    # ipython.magic("gui qt")
    ipython.run_line_magic("gui", "qt")