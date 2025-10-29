from IPython import get_ipython


def switch_to_matplotlib_qt_backend(func):  
    def wrap(*args, **kwargs):  
        ipython = get_ipython()
        ipython.run_line_magic("matplotlib", "qt")
        result = func(*args, **kwargs)
        return result
    return wrap  