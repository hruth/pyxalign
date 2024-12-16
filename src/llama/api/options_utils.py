from dataclasses import is_dataclass
import llama.api.options as opts
import llama.gpu_utils as gutils


def set_all_device_options(options, device_options: "opts.DeviceOptions"):
    "Set all device options in the input `options` dataclass to the valeus in `device_options`"

    for k, v in options.__dict__.items():
        if isinstance(v, opts.DeviceOptions):
            options.__dict__[k] = device_options
        elif is_dataclass(v):
            set_all_device_options(v, device_options)
        else:
            pass