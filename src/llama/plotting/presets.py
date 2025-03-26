import numpy as np
from llama.api.options.transform import ShiftOptions
from llama.plotting.plotters import make_image_slider_plot, ImagePlotObject
from llama.api.options.plotting import ImageSliderPlotOptions, SliderPlotOptions
from llama.transformations.classes import Shifter


def get_alignment_results_plot_object(
    data: np.ndarray,
    shift: np.ndarray,
    shift_options: ShiftOptions,
) -> list[ImagePlotObject]:

    # Make the plotting objects
    plot_objects = []
    # n_cols = 2
    shifted_data = Shifter(shift_options).run(data, shift)

    plot_objects = [
        ImagePlotObject(self.xrf_arrays[channel], options=plot_options),
        ImagePlotObject(self.xrf_arrays[channel], options=plot_options),
    ]


    for row, channel in enumerate(channels):
        plot_object = self.get_plot_object(channel)
        plot_objects += [plot_object]

        aligned_plot_object = copy.deepcopy(plot_object)
        aligned_plot_object.array = Shifter(shift_options).run(
            self.xrf_arrays[channel], self.shift
        )
        aligned_plot_object.options.slider.title = "Aligned"
        plot_objects += [aligned_plot_object]

    n_rows = len(channels)
    n_cols = 2
    make_image_slider_plot(plot_objects, (n_rows, n_cols))
    plt.show()