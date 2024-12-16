from dataclasses import dataclass
import importlib
import sys
import inspect
from typing import List

from llama.projections import ComplexProjections, PhaseProjections, Projections
from llama.api.options import ProjectionOptions

# To do: prevent overwriting! everything is imported by name which is bad!!
# Just switch to lamReload model (with submodules being reloaded) and manually re-do imports

def reload_module_and_imports(module_name="llama", namespace=None):
    """
    Reloads a module, its submodules, and dynamically re-imports all attributes
    into the specified namespace.

    The call must be `reload_module_and_imports("llama", globals())` for this to work.
    Note that using this breaks `isinstance` when objects get reloaded.

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
            # if i == 0:
                # print(module)
            for attr_name, attr_value in inspect.getmembers(module):
                if not attr_name.startswith("_"):  # Skip private attributes
                    print(attr_name)
                    namespace[attr_name] = attr_value


def refresh_projections(stale_projections: Projections) -> Projections:
    for projection_class in [PhaseProjections, ComplexProjections]:
        if projection_class.__name__ == stale_projections.__class__.__name__:
            # PhaseProjections()
            new_projections = projection_class(
                projections=stale_projections.data,
                angles=stale_projections.angles,
                options=refresh_options(stale_projections.options, ProjectionOptions),
                masks=stale_projections.masks,
                shift_manager=stale_projections.shift_manager,
                center_of_rotation=stale_projections.center_of_rotation,
                skip_pre_processing=True,
            )
            new_projections.pixel_size = stale_projections.pixel_size
    return new_projections


def refresh_options(stale_options: dataclass, options_class: dataclass) -> dataclass:
    # This won't work if fields were removed.
    return options_class(**vars(stale_options))


if __name__ == "__main__":
    from llama.api.options import DownsampleOptions
    import numpy as np

    p = PhaseProjections(
        np.zeros((3, 3, 3)),
        [1, 2, 3],
        ProjectionOptions(downsample=DownsampleOptions(scale=20)),
    )
    print("p is not stale:", isinstance(p, PhaseProjections))
    reload_module_and_imports("llama", globals())
    print("p is not stale:", isinstance(p, PhaseProjections))
    new_p = refresh_projections(p)
    print("p is not stale:", isinstance(new_p, PhaseProjections))
    print("options are not stale:", isinstance(new_p.options, ProjectionOptions))
