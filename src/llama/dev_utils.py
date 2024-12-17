from dataclasses import dataclass
import importlib
import sys
import inspect
from typing import List
from llama.api.options.task import AlignmentTaskOptions

from llama.projections import ComplexProjections, PhaseProjections, Projections
from llama.api.options.projections import ProjectionOptions
from llama.task import LaminographyAlignmentTask


def refresh_task(stale_task: LaminographyAlignmentTask) -> LaminographyAlignmentTask:
    if stale_task.complex_projections is not None:
        stale_task.complex_projections = refresh_projections(stale_task.complex_projections)
    if stale_task.phase_projections is not None:
        stale_task.phase_projections = refresh_projections(stale_task.phase_projections)

    new_task = LaminographyAlignmentTask(
        options=refresh_options(stale_task.options, AlignmentTaskOptions),
        complex_projections=stale_task.complex_projections,
        phase_projections=stale_task.phase_projections,
    )

    return new_task


def refresh_projections(stale_projections: Projections) -> Projections:
    for projection_class in [PhaseProjections, ComplexProjections]:
        if projection_class.__name__ == stale_projections.__class__.__name__:
            # PhaseProjections()
            new_projections: Projections = projection_class(
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


def reload_module_recursively(module, show_info: bool = False):
    """
    Reload a module and all its sub-modules recursively.

    Parameters:
        module: The parent module to reload (imported module object).
    """
    visited_modules = set()

    def _reload(module):
        if not module or module in visited_modules:
            return
        visited_modules.add(module)

        # Find sub-modules
        module_name = module.__name__
        submodules = [name for name in sys.modules if name.startswith(module_name + ".")]

        # Reload sub-modules first
        for sub_name in submodules:
            sub_module = sys.modules.get(sub_name)
            if sub_module:
                _reload(sub_module)

        # Reload the current module
        if show_info:
            print(f"Reloading module: {module.__name__}")
        importlib.reload(module)

    _reload(module)


if __name__ == "__main__":
    # from llama.api.options import DownsampleOptions
    # import numpy as np

    # p = PhaseProjections(
    #     np.zeros((3, 3, 3)),
    #     [1, 2, 3],
    #     ProjectionOptions(downsample=DownsampleOptions(scale=20)),
    # )
    # print("p is not stale:", isinstance(p, PhaseProjections))
    # reload_module_and_imports("llama", globals())
    # print("p is not stale:", isinstance(p, PhaseProjections))
    # new_p = refresh_projections(p)
    # print("p is not stale:", isinstance(new_p, PhaseProjections))
    # print("options are not stale:", isinstance(new_p.options, ProjectionOptions))

    # from llama.api.options import DownsampleOptions
    # import numpy as np

    # p = PhaseProjections(
    #     np.zeros((3, 3, 3)),
    #     [1, 2, 3],
    #     ProjectionOptions(downsample=DownsampleOptions(scale=20)),
    # )
    # print("p is not stale:", isinstance(p, PhaseProjections))
    # reload_module_recursively("llama")
    # print("p is not stale:", isinstance(p, PhaseProjections))
    # new_p = refresh_projections(p)
    # print("p is not stale:", isinstance(new_p, PhaseProjections))
    # print("options are not stale:", isinstance(new_p.options, ProjectionOptions))

    # reload_module_recursively
    import llama
    import llama.api.options as opts
    from llama.api import enums
    from llama.projections import PhaseProjections  # this one has to be called after the reload
    from llama.task import LaminographyAlignmentTask  # this one has to be called after the reload
    from llama.transformations.functions import image_shift_fft
    import numpy as np

    stale_gpu = opts.GPUOptions()
    stale_device = opts.DeviceOptions()
    stale_enum = enums.MemoryConfig
    stale_projections = PhaseProjections(
        np.array([[[1, 2, 3]]]), np.array([[1]]), opts.ProjectionOptions()
    )
    stale_task = LaminographyAlignmentTask(options=opts.AlignmentTaskOptions(), phase_projections=1)
    stale_func = image_shift_fft

    reload_module_recursively(llama, False)

    import llama
    import llama.api.options as opts
    from llama.api import enums
    from llama.projections import PhaseProjections  # this one has to be called after the reload
    from llama.task import LaminographyAlignmentTask  # this one has to be called after the reload
    from llama.transformations.functions import (
        image_shift_fft,
    )  # this one has to be called after the reload

    new_gpu = opts.GPUOptions()
    new_device = opts.DeviceOptions()
    new_enum = enums.MemoryConfig
    new_projections = PhaseProjections(
        np.array([[[1, 2, 3]]]), np.array([[1]]), opts.ProjectionOptions()
    )
    new_task = LaminographyAlignmentTask(options=opts.AlignmentTaskOptions(), phase_projections=1)
    new_func = image_shift_fft

    def check_staleness(name, old_object, expected_type):
        try:
            print(f"old {name} is stale:", not isinstance(old_object, expected_type))
        except:  # noqa: E722
            print(f"old {name} is stale:", old_object is not expected_type)

    print("These should all be TRUE")
    check_staleness("gpu", stale_gpu, opts.GPUOptions)
    check_staleness("device", stale_device, opts.DeviceOptions)
    check_staleness("enum", stale_enum.CPU_ONLY, new_enum)
    check_staleness("task", stale_task, type(new_task))
    check_staleness("task options attribute", stale_task.options, type(new_task.options))
    check_staleness("task options attribute", stale_task.options, opts.AlignmentTaskOptions)
    check_staleness(
        "task options device enum",
        stale_task.options.projection_matching.device.device_type,
        enums.DeviceType,
    )
    check_staleness("phase projections", stale_projections, type(new_projections))
    check_staleness(
        "phase projections laminogram",
        stale_projections.laminogram,
        type(new_projections.laminogram),
    )
    check_staleness(
        "phase projections laminogram",
        stale_projections.laminogram,
        type(new_projections.laminogram),
    )
    check_staleness("image_shift_fft", stale_func, new_func)
