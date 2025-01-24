import time
from functools import wraps
from typing import Callable, TypeVar, Optional, Union
import numpy as np
import cupy as cp
from collections import defaultdict

# Type variables to retain function signatures
T = TypeVar("T", bound=Callable)

ENABLE_TIMING = False
"Global flag to enable or disable timing."
ELAPSED_TIME_DICT: dict[str, np.ndarray] = defaultdict(lambda: np.array([]))
"""
A dictionary containing numpy arrays of the measured execution times of 
each timed function.
"""
ADVANCED_TIME_DICT: dict[str, Union[np.ndarray, dict]] = defaultdict(lambda: {})
"""
A nested dictionary, where each level of the dictionary contains
1) a key, value pair ("time": np.ndarray) that contains all measured
execution times for that function and 2) zero or more key-value 
pairs (function_name: dict) where function_name refers to the name
of each functions called in the function currently being times.

Note that only functions with the `timer` decorator will show up
in `ADVANCED_TIME_DICT`.
"""
CURRENT_DICT_REFERENCE = ADVANCED_TIME_DICT  # Initialized to the top level of ADVANCED_TIME_DICT
"""
A reference to the level of `ADVANCED_TIME_DICT` that corresponds to 
the function currently being executed.
"""
TIMING_OVERHEAD_ARRAY = np.array([])
"""
A numpy array containing measurements of how long it takes to execute 
functions in the `timer` decorator.
"""

list_of_all_gpus = tuple(range(cp.cuda.runtime.getDeviceCount()))


def toggle_timer(enable: bool):
    """
    Toggle the global ENABLE_TIMING flag.

    Parameters
    ----------
    enable : bool
        If True, enable timing. If False, disable timing.
    """
    global ENABLE_TIMING
    ENABLE_TIMING = enable


def timer(enabled: bool = True, override_with_name: Optional[str] = None):
    """
    Decorator to time a function's execution time and the execution time of the timed code
    within that function. This function is enabled or disabled depending on the state of the
    global ENABLE_TIMING flag.

    The results of the timer function will be recorded in `ELAPSED_TIME_DICT` and
    `ADVANCED_TIME_DICT`.

    Parameters
    ----------
    enabled : bool, optional
        Whether timing is enabled for the decorated function. Default is True.
    override_with_name : str, optional
        Custom name to use for the function in the timing dictionary. If not
        specified, the function name is automatically generated.

    Returns
    -------
    Callable
        The wrapped function.
    """

    def decorator(func: T) -> T:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if enabled and globals().get("ENABLE_TIMING", False):
                # Measure the overhead from running the timer function
                measure_overhead_start_1 = time.time()
                if override_with_name is None:
                    function_name = func.__qualname__
                else:
                    function_name = override_with_name
                saved_dict_reference = update_current_dict_reference(function_name)
                overhead_time_1 = time.time() - measure_overhead_start_1

                # Measure function execution time
                wait_for_process_completion_on_all_gpus()
                start_time = time.time()
                result = func(*args, **kwargs)
                wait_for_process_completion_on_all_gpus()
                elapsed_time = time.time() - start_time

                # Measure the overhead from running the timer function
                measure_overhead_start_2 = time.time()
                update_elapsed_time_dict(function_name, elapsed_time)
                update_advanced_time_dict(elapsed_time)
                # Traverse back up the advanced timing dicts
                revert_current_dict_reference(saved_dict_reference)
                global TIMING_OVERHEAD_ARRAY
                overhead_time_2 = time.time() - measure_overhead_start_2
                TIMING_OVERHEAD_ARRAY = np.append(
                    TIMING_OVERHEAD_ARRAY,
                    overhead_time_1 + overhead_time_2,
                )
            else:
                # If timing is disabled, just call the function
                result = func(*args, **kwargs)
            return result

        # Ensure the wrapper function has the same type as the original
        return wrapper  # type: ignore

    return decorator


class InlineTimer:
    """
    A timer class for inline timing of code blocks.

    Parameters
    ----------
    name : str
        The name associated with the timer that will be recorded
        in the timing dictionaries.
    enabled : bool, optional
        Whether the timer is enabled, by default True.
    """

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.overhead_time = 0

    def start(self):
        """
        Starts the timer if timing is enabled.
        """
        if self.enabled and globals().get("ENABLE_TIMING", False):
            measure_overhead_start = time.time()
            saved_dict_reference = update_current_dict_reference(self.name)
            self.saved_dict_reference = saved_dict_reference
            self.overhead_time = time.time() - measure_overhead_start
            wait_for_process_completion_on_all_gpus()
            self.start_time = time.time()

    def end(self):
        """
        Stops the timer and records the elapsed time if timing is enabled.
        """
        if self.enabled and globals().get("ENABLE_TIMING", False):
            wait_for_process_completion_on_all_gpus()
            elapsed_time = time.time() - self.start_time
        
            measure_overhead_start = time.time()
            update_elapsed_time_dict(self.name, elapsed_time)
            update_advanced_time_dict(elapsed_time)
            revert_current_dict_reference(self.saved_dict_reference)
            global TIMING_OVERHEAD_ARRAY
            self.overhead_time += time.time() - measure_overhead_start
            TIMING_OVERHEAD_ARRAY = np.append(TIMING_OVERHEAD_ARRAY, self.overhead_time)


def update_elapsed_time_dict(function_name: str, elapsed_time: float):
    """
    Updates the global elapsed time dictionary with the elapsed time for a function.

    Parameters
    ----------
    function_name : str
        The name of the function being timed.
    elapsed_time : float
        The elapsed time for the function execution.
    """
    ELAPSED_TIME_DICT[function_name] = np.append(ELAPSED_TIME_DICT[function_name], elapsed_time)


def update_current_dict_reference(function_name: str) -> dict:
    """
    Updates the current reference in the advanced timing dictionary to a nested level.

    Parameters
    ----------
    function_name : str
        The name of the function being timed.

    Returns
    -------
    dict
        The previous dictionary reference.
    """
    global CURRENT_DICT_REFERENCE
    # Save the parent to traverse back to later
    saved_dict_reference = CURRENT_DICT_REFERENCE
    # Create new dict if necessary
    if function_name not in CURRENT_DICT_REFERENCE.keys():
        CURRENT_DICT_REFERENCE[function_name] = defaultdict(lambda: {})
        CURRENT_DICT_REFERENCE[function_name]["time"] = np.array([])
    # Update the pointer to the current dict
    CURRENT_DICT_REFERENCE = CURRENT_DICT_REFERENCE[function_name]
    return saved_dict_reference


def update_advanced_time_dict(elapsed_time: float):
    """
    Updates the advanced timing dictionary with the elapsed time.

    Parameters
    ----------
    elapsed_time : float
        The elapsed time for the function execution.
    """
    global CURRENT_DICT_REFERENCE
    CURRENT_DICT_REFERENCE["time"] = np.append(CURRENT_DICT_REFERENCE["time"], elapsed_time)


def revert_current_dict_reference(saved_dict_reference: dict):
    """
    Reverts the current dictionary reference in the advanced timing dictionary.

    Parameters
    ----------
    saved_dict_reference : dict
        The saved dictionary reference to revert to.
    """
    global CURRENT_DICT_REFERENCE
    CURRENT_DICT_REFERENCE = saved_dict_reference


# def clear_timer_globals():
def clear_timer_globals():
    """
    Clears the global timing dictionaries and resets the state.
    """
    global ELAPSED_TIME_DICT
    global ADVANCED_TIME_DICT
    global CURRENT_DICT_REFERENCE
    global TIMING_OVERHEAD_ARRAY
    ELAPSED_TIME_DICT = defaultdict(lambda: np.array([]))
    ADVANCED_TIME_DICT = defaultdict(lambda: {})
    CURRENT_DICT_REFERENCE = ADVANCED_TIME_DICT
    TIMING_OVERHEAD_ARRAY = np.array([])


def wait_for_process_completion_on_all_gpus():
    for i in list_of_all_gpus:
        with cp.cuda.Device(i):
            cp.cuda.Stream.null.synchronize()
