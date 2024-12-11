import time
from functools import wraps
from typing import Callable, TypeVar, Optional

# Type variables to retain function signatures
T = TypeVar("T", bound=Callable)

# Global flag to enable or disable timing
ENABLE_TIMING = True

tabs = ""

# This function should NOT be used to wrap any functions that will be "chunked"
def timer(prefix: str = ""):
    """Decorator to time a function and print the function name if ENABLE_TIMING is True."""
    if prefix != "":
        prefix = prefix + "."
    def decorator(func: T) -> T:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if ENABLE_TIMING:
                global tabs
                spaces = 5
                print(f"{tabs}Running function '{prefix}{func.__name__}'...")
                tabs += " " * spaces
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                tabs = tabs[:-spaces]
                print(f"{tabs}Function '{prefix}{func.__name__}': {elapsed_time:.4f} seconds")
                return result
            else:
                # If timing is disabled, just call the function
                return func(*args, **kwargs)

        # Ensure the wrapper function has the same type as the original
        return wrapper  # type: ignore

    return decorator


# import time
# from functools import wraps
# from typing import Callable, TypeVar

# # Type variables to retain function signatures
# T = TypeVar("T", bound=Callable)

# # Global flag to enable or disable timing
# ENABLE_TIMING = True

# tabs = ""

# def timer(func: T) -> T:
#     """Decorator to time a function and print the class and function name if ENABLE_TIMING is True."""
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         if ENABLE_TIMING:
#             global tabs
#             spaces = 5
#             tabs += " " * spaces

#             # Determine the class name if the function is a method
#             class_name = ""
#             if args and hasattr(args[0], "__class__"):
#                 class_name = f"{args[0].__class__.__name__}."

#             start_time = time.time()
#             result = func(*args, **kwargs)
#             elapsed_time = time.time() - start_time
#             tabs = tabs[:-spaces]
#             print(f"{tabs}Function '{class_name}{func.__name__}': {elapsed_time:.4f} seconds")
#             return result
#         else:
#             # If timing is disabled, just call the function
#             return func(*args, **kwargs)

#     return wrapper  # type: ignore

# # Example usage
# class ExampleClass:
#     @timer
#     def example_method(self, n: int) -> str:
#         """Simulates a time-consuming method."""
#         time.sleep(n)
#         return f"Waited for {n} seconds"
