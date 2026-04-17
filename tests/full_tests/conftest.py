import pytest
from typing import Dict, Callable, List, Optional

# Global registry for processing functions
PROCESSING_FUNCTIONS_REGISTRY: Dict[str, Callable] = {}
# Registry for test names (used during collection)
TEST_NAMES_REGISTRY: Dict[str, List[str]] = {}


def register_processing_function(name: str, test_names: Optional[List[str]] = None):
    """Decorator to register processing functions

    Args:
        name: Name of the processing function
        test_names: Optional list of test names that will be generated.
                   If provided, these names will be used during --collect-only
                   without executing the function.
    """
    def decorator(func: Callable):
        PROCESSING_FUNCTIONS_REGISTRY[name] = func
        if test_names is not None:
            TEST_NAMES_REGISTRY[name] = test_names
        return func
    return decorator


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Hook to dynamically generate test parameters"""
    if (
        "test_name" in metafunc.fixturenames
        and "result" in metafunc.fixturenames
        and metafunc.function.__name__ == "test_single_result"
    ):
        # Check if we're in collect-only mode
        if metafunc.config.option.collectonly:
            # During collection, just create placeholder parameters
            # This prevents the actual test functions from executing
            all_test_results = [(f"{func_name}::placeholder", True)
                                for func_name in PROCESSING_FUNCTIONS_REGISTRY.keys()]
        else:
            all_test_results = []

            for func_name, func in PROCESSING_FUNCTIONS_REGISTRY.items():
                test_results = func()
                for name, value in test_results.items():
                    prefixed_name = f"{func_name}::{name}"
                    # prefixed_name = name
                    all_test_results.append((prefixed_name, value))

        metafunc.parametrize(
            "test_name,result",
            all_test_results,
            ids=[name for name, _ in all_test_results],
        )