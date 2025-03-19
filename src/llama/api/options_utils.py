from dataclasses import is_dataclass
import llama.api.options as opts


def set_all_device_options(options, device_options: "opts.DeviceOptions"):
    "Set all device options in the input `options` dataclass to the valeus in `device_options`"

    for k, v in options.__dict__.items():
        # I am not using isinstance here because it does not always work as expectd when
        # using the reload_module_recursively and refresh_task function.
        if is_dataclass(v) and v.__class__.__qualname__ == opts.DeviceOptions.__qualname__:
            options.__dict__[k] = device_options
        elif is_dataclass(v):
            set_all_device_options(v, device_options)
        else:
            pass


def print_options(options, prepend="- "):
    for k, v in options.__dict__.items():
        if is_any_dataclass_instance(v):
            print(f"{prepend}{k} options")
            print_options(v, "     " + prepend)
        else:
            print(f"{prepend}{k}: {v}")


def is_any_dataclass_instance(obj):
    return is_dataclass(obj) and not isinstance(obj, type)
