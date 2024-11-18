from ipywidgets import interact
import ipywidgets as widgets

import matplotlib.pyplot as plt
import numpy as np

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import interact

from llama.api.types import ArrayType


def make_image_slider_plot(images: ArrayType):
    # Create the play button and slider (slider will not be displayed)
    play = widgets.Play(
        value=0,
        min=0,
        max=len(images) - 1,
        interval=500,
        description="Play",
        disabled=False,
    )

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(images) - 1,
        description="index",
        visible=False,  # Hide the slider
    )

    # Link the play button and slider
    widgets.jslink((play, "value"), (slider, "value"))

    # Function to update the plot
    def update_plot(idx):
        plt.imshow(images[idx])
        plt.show()

    # Use interact with the slider (without displaying the slider)
    interact(update_plot, idx=slider)

    # Display only the play button
    display(play)


def plot_sum_of_images(images: ArrayType):
    plt.imshow(images.sum(0))
    plt.show()
