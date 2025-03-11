import numpy as np
from ipywidgets import interact
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display


class ThresholdSelector:
    is_final_value = False

    def __init__(self, masks, projections):
        self.masks = masks
        self.projections = projections

    # Function to update the plot
    def update_plot(self, idx, thresh):
        self.threshold = thresh
        fig, ax = plt.subplots(1, 3, layout="compressed")
        clipped_masks = self.masks[idx] * 1
        clip_idx = clipped_masks > thresh
        clipped_masks[:] = 0
        clipped_masks[clip_idx] = 1
        # clipped_masks[clipped_masks > thresh] = 1
        # clipped_masks[clipped_masks < thresh] = 0
        plt.sca(ax[0])
        plt.imshow(clipped_masks)
        plt.title("Mask")
        plt.clim([0, 1])
        plt.sca(ax[1])
        plt.imshow(clipped_masks * np.angle(self.projections[idx]))
        plt.title(r"Mask $\times$ Projection")
        plt.sca(ax[2])
        plt.imshow((1 - clipped_masks) * np.angle(self.projections[idx]))
        plt.title(r"(1-Mask) $\times$ Projection")
        plt.show()


def illum_map_threshold_plotter(
    masks: np.ndarray, projections: np.ndarray, init_thresh: float
) -> ThresholdSelector:
    # Create the play button and slider
    play = widgets.Play(
        value=0,
        min=0,
        max=len(masks) - 1,
        interval=500,
        description="Play",
    )

    index_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(masks) - 1,
        description="index",
        visible=False,  # The slider will be shown via the interact link
    )

    thresh_float_text = widgets.BoundedFloatText(
        value=init_thresh,
        min=0,
        description="threshold",
        visible=False,  # The slider will be shown via the interact link
    )

    # Stop button
    stop_button = widgets.Button(
        description="Select this threshold value",
        button_style="danger",
        tooltip="Stop interactivity and return threshold value",
    )

    output = widgets.Output()

    # class ThresholdSelector:
    #     is_final_value = False

    #     # Function to update the plot
    #     def update_plot(self, idx, thresh):
    #         self.thresh = thresh
    #         fig, ax = plt.subplots(1, 3, layout="compressed")
    #         clipped_masks = masks[idx] * 1
    #         clipped_masks[clipped_masks > thresh] = 1
    #         clipped_masks[clipped_masks < thresh] = 0
    #         plt.sca(ax[0])
    #         plt.imshow(clipped_masks)
    #         plt.title("Mask")
    #         plt.clim([0, 1])
    #         plt.sca(ax[1])
    #         plt.imshow(clipped_masks * np.angle(projections[idx]))
    #         plt.title(r"Mask $\times$ Projection")
    #         plt.sca(ax[2])
    #         plt.imshow((1 - clipped_masks) * np.angle(projections[idx]))
    #         plt.title(r"(1-Mask) $\times$ Projection")
    #         plt.show()

    # Link the play button and slider
    widgets.jslink((play, "value"), (index_slider, "value"))

    threshold_selector = ThresholdSelector(masks, projections)
    interact(
        threshold_selector.update_plot,
        idx=index_slider,
        thresh=thresh_float_text,
    )

    # Stop button callback function
    def stop_interaction(b):
        threshold_selector.is_final_value = True
        del threshold_selector.masks
        del threshold_selector.projections
        play.disabled = True
        index_slider.disabled = True
        thresh_float_text.disabled = True
        stop_button.disabled = True  # Disable itself
        with output:
            output.clear_output()
            print(f"Final threshold value: {thresh_float_text.value}")

    stop_button.on_click(stop_interaction)

    display(play, stop_button, output)

    return threshold_selector
