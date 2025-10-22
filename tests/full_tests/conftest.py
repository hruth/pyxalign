import pytest
from typing import Dict, Callable

# Global registry for processing functions
PROCESSING_FUNCTIONS_REGISTRY: Dict[str, Callable] = {}


def register_processing_function(name: str):
    """Decorator to register processing functions"""
    def decorator(func: Callable):
        PROCESSING_FUNCTIONS_REGISTRY[name] = func
        return func
    return decorator


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Hook to dynamically generate test parameters"""
    if (
        "test_name" in metafunc.fixturenames
        and "result" in metafunc.fixturenames
        and metafunc.function.__name__ == "test_single_result"
    ):
        all_test_results = []
        
        for func_name, func in PROCESSING_FUNCTIONS_REGISTRY.items():
            test_results = func()
            for name, value in test_results.items():
                # prefixed_name = f"{func_name}::{name}"
                prefixed_name = name
                all_test_results.append((prefixed_name, value))
        
        metafunc.parametrize(
            "test_name,result",
            all_test_results,
            ids=[name for name, _ in all_test_results],
        )