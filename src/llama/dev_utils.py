import importlib
import sys
import inspect

def reload_module_and_imports(module_name, namespace=None):
    """
    Reloads a module, its submodules, and dynamically re-imports all attributes
    into the specified namespace.

    The call must be reload_module_and_imports("llama", globals()) for this to work.

    Args:
        module_name (str): The top-level module name to reload.
        namespace (dict): The namespace to update with reloaded attributes. Defaults to globals().
    """
    if module_name not in sys.modules:
        raise ImportError(f"Module '{module_name}' is not currently loaded.")
    
    # Default to the global namespace of the caller
    if namespace is None:
        namespace = globals()

    # Collect all submodules
    modules_to_reload = [name for name in sys.modules if name.startswith(module_name)]
    
    # Some things aren't updated on the first pass, so I reload way too many
    # times to ensure everything is reloaded properly.
    num_reloads = 10
    for i in range(num_reloads):
        # Reload all modules in reverse order (to handle dependencies correctly)
        for name in sorted(modules_to_reload, key=len, reverse=True):
            importlib.reload(sys.modules[name])
        
        # Re-import all attributes from submodules into the specified namespace
        for name in modules_to_reload:
            module = sys.modules[name]
            for attr_name, attr_value in inspect.getmembers(module):
                if not attr_name.startswith("_"):  # Skip private attributes
                    namespace[attr_name] = attr_value