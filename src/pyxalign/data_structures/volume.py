from typing import Optional, Sequence
from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
import astra
import copy
import h5py
from PyQt5.QtWidgets import QApplication

from pyxalign.api.constants import divisor
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.options.plotting import PlotDataOptions
from pyxalign.api.options.transform import RotationOptions
from pyxalign.gpu_utils import create_empty_pinned_array_like, get_scipy_module, pin_memory

import pyxalign.image_processing as ip
from pyxalign import reconstruct
from pyxalign.io.save import save_array_as_tiff
from pyxalign.plotting.interactive.arrays import VolumeViewer
from pyxalign.plotting.interactive.launchers import launch_volume_viewer
from pyxalign.plotting.plotters import plot_slice_of_3D_array
import pyxalign.data_structures.projections as projections
from pyxalign.timing.timer_utils import timer
import matplotlib.pyplot as plt

from pyxalign.api.types import ArrayType, r_type
from pyxalign.transformations.classes import Rotator
from pyxalign.transformations.functions import image_rotate_fft


class Volume:
    def __init__(
        self,
        projections: "projections.PhaseProjections",
    ):
        # Store a reference to the projections
        self.projections = projections
        self.options = projections.options.reconstruct
        self.experiment_options = projections.options.experiment

        self.astra_config: dict = None
        self.object_geometries: dict = None
        self.scan_geometry_config: dict = None
        self.vectors: np.ndarray = None
        self.forward_projection_id: int = None
        self.is_initialized: bool = False
        self.optimal_rotation_angles: Sequence[float] = [0, 0, 0]
        self.data = None
        self.forward_projections = None

    @property
    def n_layers(self):
        return self.projections.reconstructed_object_dimensions[2]

    def intialize_astra_reconstructor_inputs(self, n_pix: Optional[Sequence[int]] = None):
        if n_pix is None:
            n_pix = self.projections.reconstructed_object_dimensions
        scan_geometry_config, vectors = reconstruct.get_astra_reconstructor_geometry(
            size=self.projections.size,
            angles=self.projections.angles,
            n_pix=n_pix,
            center_of_rotation=self.projections.center_of_rotation,
            lamino_angle=self.experiment_options.laminography_angle,
            tilt_angle=self.options.geometry.tilt_angle,
            skew_angle=self.options.geometry.skew_angle,
        )
        object_geometries = reconstruct.get_object_geometries(scan_geometry_config, vectors)
        return scan_geometry_config, vectors, object_geometries

    @timer()
    def generate_volume(
        self,
        filter_inputs: bool = True,
        pinned_filtered_sinogram: Optional[np.ndarray] = None,
        reinitialize_astra: bool = True,
        n_pix: Optional[Sequence[int]] = None,
        update_stored_sinogram: bool = True,
        update_geometries: bool = False,
    ):
        # reinforce references
        self.options = self.projections.options.reconstruct
        self.experiment_options = self.projections.options.experiment

        # Copy the settings used at the time of the reconstruction
        # self.options = copy.deepcopy(self.projections.options.reconstruct)
        # self.experiment_options = copy.deepcopy(self.projections.options.experiment)
        device = cp.cuda.Device()
        # Re-initialize the inputs and clear outputs
        if reinitialize_astra or not self.is_initialized:
            self.clear_astra_objects()
            self.scan_geometry_config, self.vectors, self.object_geometries = (
                self.intialize_astra_reconstructor_inputs(n_pix=n_pix)
            )
            self.is_initialized = True
        elif update_geometries and not reinitialize_astra:
            # Re-initialize geometries, but not the whole astra object
            self.scan_geometry_config, self.vectors, self.object_geometries = (
                self.intialize_astra_reconstructor_inputs(n_pix=n_pix)
            )
            # update the geometries, but do not update the stored projections
            reconstruct.update_astra_reconstructor_config(
                self.object_geometries,
                self.astra_config,
            )

        if update_stored_sinogram:
            # Prepare the projections before doing the back-projection
            # if (sinogram is None) and update_stored_sinogram:
            if filter_inputs:
                if pinned_filtered_sinogram is None:
                    pinned_filtered_sinogram = create_empty_pinned_array_like(self.projections.data)
                sinogram = reconstruct.filter_sinogram(
                    sinogram=self.projections.data,
                    vectors=self.vectors,
                    device_options=self.options.filter.device,
                    pinned_results=pinned_filtered_sinogram,
                )
            else:
                sinogram = self.projections.data

        if self.astra_config is None:
            # allocate memory for volume and projections, and store projections
            self.astra_config = reconstruct.create_astra_reconstructor_config(
                sinogram,
                self.object_geometries,
                self.options.astra.algorithm_type,
            )
        else:
            if update_stored_sinogram:
                # update the stored projections
                reconstruct.update_stored_sinogram(sinogram, self.astra_config)

        # size of the 3D reconstruction
        volume_shape = np.array(
            [
                self.scan_geometry_config["iVolZ"],
                self.scan_geometry_config["iVolX"],
                self.scan_geometry_config["iVolY"],
            ]
        )
        # device = cp.cuda.Device()
        astra.set_gpu_index(self.options.astra.back_project_gpu_indices)
        cp.cuda.Device(device).use()
        if self.data is None or not np.all(self.data.shape == volume_shape):
            self.data = reconstruct.get_3D_reconstruction(self.astra_config)
        else:
            self.data[:] = reconstruct.get_3D_reconstruction(self.astra_config)
        cp.cuda.Device(device).use()

    @timer()
    def get_forward_projection(
        self,
        pinned_forward_projection: Optional[np.ndarray] = None,
        forward_projection_id: Optional[int] = None,
    ):
        # If no forward_projection_id was passed in, a new projection
        # id will be generated. If that is the case, save that ID so
        # you can skip the creation of a new 3D astra object next time
        # this is run
        is_new_sino_id_created = (forward_projection_id is None) and (
            self.forward_projection_id is None
        )

        if forward_projection_id is None:
            forward_projection_id = self.forward_projection_id

        device = cp.cuda.Device()
        astra.set_gpu_index(self.options.astra.forward_project_gpu_indices)
        cp.cuda.Device(device).use()
        forward_projections, forward_projection_id = reconstruct.get_forward_projection(
            reconstruction=self.data,  # should remove this since volume id is always provided
            object_geometries=self.object_geometries,
            pinned_forward_projection=pinned_forward_projection,
            volume_id=self.astra_config["ReconstructionDataId"],
            forward_projection_id=forward_projection_id,
            return_id=True,
        )
        cp.cuda.Device(device).use()

        if is_new_sino_id_created:
            self.forward_projection_id = forward_projection_id

        # Create a projections object in case I want to use any of the
        # projections object methods
        forward_projection_options = copy.deepcopy(self.projections.options)
        forward_projection_options.experiment.pixel_size = self.projections.pixel_size
        self.forward_projections = projections.PhaseProjections(
            projections=forward_projections,
            options=forward_projection_options,
            angles=self.projections.angles,
            scan_numbers=self.projections.scan_numbers,
            center_of_rotation=self.projections.center_of_rotation,
            skip_pre_processing=True,
        )

    def generate_projection_masks_from_circulo(
        self,
        forward_project_gpu_indices: Optional[tuple] = None,
        radial_smooth: int = 0,
        rad_apod: int = 0,
    ) -> np.ndarray:
        device = cp.cuda.Device()
        if forward_project_gpu_indices is None:
            astra.set_gpu_index(self.options.astra.forward_project_gpu_indices)
        else:
            astra.set_gpu_index(forward_project_gpu_indices)
        cp.cuda.Device(device).use()

        reconstruction_mask = self.get_circular_window(radial_smooth, rad_apod)
        reconstruction_mask = np.repeat(reconstruction_mask[None], self.n_layers, axis=0)

        _, _, object_geometries = self.intialize_astra_reconstructor_inputs()
        mask, sino_id = reconstruct.get_forward_projection(
            reconstruction=reconstruction_mask,
            object_geometries=object_geometries,
            return_id=True,
        )
        mask = mask[0] / mask[0].max()
        astra.data3d.delete(sino_id)

        return mask

    def clear_astra_objects(self):
        astra_objects = []
        if self.astra_config is not None:
            astra_objects += [
                self.astra_config["ReconstructionDataId"],
                self.astra_config["ProjectionDataId"],
            ]
        if self.forward_projection_id is not None:
            astra_objects += [self.forward_projection_id]
        astra.data3d.delete(astra_objects)

        self.astra_config = None
        self.object_geometries = None
        self.scan_geometry_config = None
        self.vectors = None
        self.forward_projection_id = None

    @timer()
    def apply_circular_window(self, circulo: Optional[ArrayType] = None):
        if circulo is None:
            self.data[:] = self.data * self.get_circular_window()
        else:
            self.data[:] = self.data * circulo

    def get_circular_window(self, radial_smooth: int, rad_apod: int):
        # was 5 and 0
        # Generate circular mask for reconstruction
        return ip.apply_3D_apodization(
            image=np.zeros(self.projections.reconstructed_object_dimensions),
            rad_apod=rad_apod,
            radial_smooth=radial_smooth,
        ).astype(r_type)

    def plot_data(
        self,
        options: Optional[PlotDataOptions] = None,
        show_plot: bool = True,
    ):
        if options is None:
            options = PlotDataOptions()
        else:
            options = copy.deepcopy(options)

        if options.index is None:
            options.index = int(self.data.shape[0] / 2)

        plt.title(f"Reconstruction Slice {options.index}")
        plot_slice_of_3D_array(
            self.data,
            options,
            self.projections.pixel_size,
            show_plot=show_plot,
        )

    def get_optimal_rotation_of_reconstruction(
        self,
        use_gpu: bool = True,
        slice_index: Optional[int] = None,
        pad_mult: int = 4,
    ):
        self.optimal_rotation_angles = get_tomogram_rotation_angles(
            self.data, use_gpu, slice_index, pad_mult
        )
        print(
            "Optimal rotation values:\n"
            + f"\tx: {self.optimal_rotation_angles[0]}\n"
            + f"\ty: {self.optimal_rotation_angles[1]}\n"
            + f"\tz: {self.optimal_rotation_angles[2]}"
        )

    def rotate_reconstruction(self, device_options: Optional[DeviceOptions] = None) -> np.ndarray:
        if device_options is None:
            device_options = DeviceOptions()

        rotation_options = RotationOptions(device=device_options, enabled=True)
        rotator = Rotator(rotation_options)

        rotated_reconstruction = pin_memory(self.data)

        ax = [1, 0, 2]
        rotator.options.angle = self.optimal_rotation_angles[0]
        rotated_reconstruction = rotator.run(self.data.transpose(ax)).transpose(ax)

        ax = [2, 1, 0]
        rotator.options.angle = -self.optimal_rotation_angles[1]
        rotated_reconstruction = rotator.run(rotated_reconstruction.transpose(ax)).transpose(ax)

        rotator.options.angle = self.optimal_rotation_angles[2]
        rotated_reconstruction = rotator.run(rotated_reconstruction)

        print(
            "Rotated reconstruction by:\n"
            + f"\tx: {self.optimal_rotation_angles[0]}\n"
            + f"\ty: {self.optimal_rotation_angles[1]}\n"
            + f"\tz: {self.optimal_rotation_angles[2]}"
        )
        self.optimal_rotation_angles = [0, 0, 0]
        self.data = rotated_reconstruction

    def save_as_tiff(
        self,
        file_path: str,
        min: Optional[float] = None,
        max: Optional[float] = None,
        data: Optional[np.ndarray] = None,
    ):
        if data is None and self.data is None:
            print("There is no volume data to save!")
        if data is None:
            data = self.data

        save_array_as_tiff(data, file_path, min, max)

    def save_as_h5(self, file_path: str):
        if self.data is None:
            print("There is no volume data to save!")
        with h5py.File(file_path, "w") as F:
            F.create_dataset(name="volume", data=self.data)

    def launch_viewer(self):
        self.gui = launch_volume_viewer(self.data)


def get_tomogram_rotation_angles(
    reconstruction: np.ndarray,
    use_gpu: bool = True,
    slice_index: Optional[int] = None,
    pad_mult: int = 4,
):
    if use_gpu:
        xp = cp
    else:
        xp = np
    if slice_index is None:
        slice_index = int(reconstruction.shape[1] / 2)
    rotation_angle = np.zeros(3, dtype=r_type)

    for i in range(3):
        if i == 2:
            reconstruction_slice = reconstruction.mean(axis=0)
            max_search_angle = 22.5
        else:
            reconstruction_slice = reconstruction.take(indices=slice_index, axis=i + 1)
            # Assume the sample is properly centered and calculate
            # the max search angle
            h, w = reconstruction_slice.shape
            max_search_angle = np.arctan(2 * h / w) * 180 / np.pi
            pad_mult = np.ceil(reconstruction_slice.shape[0] / divisor)
            pad_by = int(pad_mult * divisor)
            reconstruction_slice = np.pad(reconstruction_slice, ((pad_by, pad_by), (0, 0)))
        rotation_angle[i] = get_optimized_sparseness_angle(
            xp.array(reconstruction_slice),
            angle_search_bounds=[-max_search_angle, max_search_angle],
        )

    return rotation_angle


def get_optimized_sparseness_angle(
    image_slice: ArrayType, angle_search_bounds: Sequence, n_iter: int = 500
):
    """
    Find the rotation of the object that maximizes sparsity
    """
    xp = cp.get_array_module(image_slice)
    scipy_module = get_scipy_module(image_slice)

    def get_hoyer_sparsity(image: ArrayType):
        # image = image[:]
        sqrt_n = xp.sqrt(len(image))
        l1_norm = xp.linalg.norm(image, ord=1)
        l2_norm = xp.linalg.norm(image, ord=2)
        sparsity = (sqrt_n - l1_norm / l2_norm) / (sqrt_n - 1)

        return sparsity

    # generate circular mask for image
    n_pix = np.shape(image_slice)
    [X, Y] = xp.meshgrid(
        xp.arange(-np.ceil(n_pix[1] / 2), np.floor(n_pix[1] / 2), dtype=r_type),
        xp.arange(-np.ceil(n_pix[0] / 2), np.floor(n_pix[0] / 2), dtype=r_type),
    )
    elliptical_mask = X**2 / (n_pix[1] / 2) ** 2 + Y**2 / (n_pix[0] / 2) ** 2 < 1 / 2

    @timer()
    def get_image_sparsity_score(image: ArrayType, angle: float):
        image = image * elliptical_mask
        image = image - scipy_module.ndimage.gaussian_filter(image, sigma=5)
        image = image_rotate_fft(image[None], angle)[0]
        # Remove edges
        image = image[
            np.ceil(image.shape[0] * 0.1) : np.floor(image.shape[0] * 0.9),
            np.ceil(image.shape[1] * 0.1) : np.floor(image.shape[1] * 0.9),
        ]
        image = xp.abs(scipy_module.fft.fftshift(scipy_module.fft.fft2(image)))
        score = xp.array(
            [
                get_hoyer_sparsity(image.mean(axis=0)),
                get_hoyer_sparsity(image.mean(axis=1)),
            ]
        )
        score = -xp.mean(score)
        # score = -score
        if xp is cp:
            return score.get()
        else:
            return score

    test_image = image_slice
    # test_image = xp.abs(image_slice)
    # test_image = test_image - xp.median(test_image)
    # test_image[test_image < 0] = 0

    def get_score_vs_angle(image, angles: np.ndarray):
        score = []
        for i in range(len(angles)):
            score += [get_image_sparsity_score(image, angles[i])]
        return np.array(score, dtype=r_type)

    angles = np.linspace(angle_search_bounds[0], angle_search_bounds[-1], n_iter)
    next_range = np.array([-1, 1])

    # Do grid search of the sparsity score and plot results
    # fig, ax = plt.subplots(2, 2, layout="compressed")
    fig = plt.figure(layout="compressed")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])
    sparsity_axis = fig.add_subplot(gs[0, 0:2])
    orig_slice_axis = fig.add_subplot(gs[1, 0])
    rotated_slice_axis = fig.add_subplot(gs[1, 1])

    # Find and plot the sparsity score
    plt.sca(sparsity_axis)
    plt.title("Hoyer Sparsity Score")
    plt.xlabel("angle (deg)")
    plt.grid(linestyle=":")
    plt.autoscale(enable=True, axis="x", tight=True)
    for i in range(3):
        score = get_score_vs_angle(test_image, angles)
        plt.plot(angles, score)
        angles = angles[np.argmin(score)] + np.arange(
            next_range[0], next_range[1], next_range[1] / 10
        )
        next_range = next_range / 10
    angle = angles[np.argmin(score)]

    # Plot the image slice
    plt.title("Original slice")
    plt.sca(orig_slice_axis)
    if isinstance(image_slice, cp.ndarray):
        plt.imshow(image_slice.get(), cmap="bone")
    else:
        plt.imshow(image_slice, cmap="bone")
    # Plot the rotated image slice
    plt.sca(rotated_slice_axis)
    rotated_image_slice = image_rotate_fft(image_slice[None], angle)[0]
    if isinstance(rotated_image_slice, cp.ndarray):
        rotated_image_slice = rotated_image_slice.get()
    plt.imshow(rotated_image_slice, cmap="bone")
    plt.show()

    return angle
