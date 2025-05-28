from typing import Sequence, Union
import h5py
import numpy as np
import dataclasses
from enum import StrEnum
from numbers import Number
from pyxalign.api.enums import SpecialValuePlaceholder
import tifffile as tiff
import os
from PIL import Image, ImageDraw, ImageFont


def save_generic_data_structure_to_h5(d: dict, h5_obj: Union[h5py.Group, h5py.File]):
    "Create h5 datasets for all items in the passed in dict"
    # This will need to be updated any time you are adding variables whose type
    # doesn't correspond to any of the if/else statements
    if not isinstance(d, dict):
        d = d.__dict__
    for value_name, value in d.items():
        if dataclasses.is_dataclass(value):
            # Recursively handle dataclasses
            save_generic_data_structure_to_h5(value, h5_obj.create_group(value_name))

        elif isinstance(value, list) and len(value) == 0:
            # Empty lists
            h5_obj.create_dataset(value_name, data=SpecialValuePlaceholder.EMPTY_LIST._value_)

        elif value is None:
            # None types
            h5_obj.create_dataset(value_name, data=SpecialValuePlaceholder.NONE._value_)

        elif isinstance(value, Union[bool, np.bool_]):
            h5_obj.create_dataset(value_name, data=value)

        elif isinstance(value, np.ndarray):
            # Arrays
            h5_obj.create_dataset(value_name, data=value, dtype=value.dtype)

        elif isinstance(value, Number):
            # Individual numbers
            h5_obj.create_dataset(value_name, data=value, dtype=type(value))

        elif isinstance(value, Sequence) and isinstance(value[0], Number):
            # Sequence (i.e. list, tuple) of numbers
            h5_obj.create_dataset(value_name, data=value, dtype=type(value[0]))

        elif isinstance(value, str):
            # Strings and string enums
            save_string_to_h5(h5_obj, value, value_name)

        elif isinstance(value, list) and (len(value) > 0 and isinstance(value[0], str)):
            # List of string enums
            sub_group = h5_obj.create_group(value_name)
            for i, list_entry in enumerate(value):
                save_string_to_h5(sub_group, list_entry, str(i))

        elif isinstance(value, list) and (len(value) > 0 and isinstance(value[0], np.ndarray)):
            # List of numpy arrays
            sub_group = h5_obj.create_group(value_name)
            for i, list_entry in enumerate(value):
                sub_group.create_dataset(str(i), data=list_entry, dtype=list_entry.dtype)

        else:
            print(f"WARNING: {value_name} not saved")


def save_string_to_h5(h5_obj: Union[h5py.Group, h5py.File], string: str, value_name: str):
    if isinstance(string, StrEnum):
        h5_obj.create_dataset(value_name, data=string._value_)
    else:
        h5_obj.create_dataset(value_name, data=string)


def save_options_to_h5_file(file_path: str, options):
    F = h5py.File(file_path, "w")
    save_generic_data_structure_to_h5(options, F)
    F.close()


def convert_to_uint_16(images: np.ndarray, min: float = None, max: float = None):
    images = images.copy()
    if min is None:
        min = images.min()
    if max is None:
        max = images.max()
    delta = max - min
    images[images < min] = min
    images[images > max] = max
    return (65535 * (images - min) / delta).astype(np.uint16)


def draw_text_with_default_font(
    base_image: Image.Image,
    text: str,
    x: int,
    y: int,
    scale_factor: float = 2.0,
) -> None:
    """
    Draw text at position (x, y) using Pillow's default bitmap font,
    scaled up by 'scale_factor'. The drawing is done *in place* on base_image.

    :param base_image: Pillow Image in mode "I" (32-bit integer) or similar.
    :param text: The text string to draw.
    :param x: Left coordinate on the base image where text will be placed.
    :param y: Top coordinate on the base image where text will be placed.
    :param scale_factor: How much to scale up the default font's size.
    """
    # 1) Create a temporary image (grayscale "L") to hold the text at default size
    #    Use a generous size so the text won’t get clipped.
    tmp_w, tmp_h = 300, 100
    tmp_img = Image.new("L", (tmp_w, tmp_h), 0)
    draw_tmp = ImageDraw.Draw(tmp_img)

    # 2) Draw text in white (255) at the top-left corner of tmp_img
    default_font = ImageFont.load_default()
    draw_tmp.text((0, 0), text, fill=255, font=default_font)

    # 3) Crop the temporary image to the bounding box of the actual drawn text
    bbox = tmp_img.getbbox()
    if bbox is None:
        # No bounding box found (if the text is empty), so no drawing
        return
    cropped = tmp_img.crop(bbox)

    # 4) Scale (resize) the cropped text image
    scaled_w = int(cropped.width * scale_factor)
    scaled_h = int(cropped.height * scale_factor)
    scaled_text_img = cropped.resize((scaled_w, scaled_h), Image.NEAREST)

    # 5) Paste the scaled text onto the base image using scaled_text_img as a mask.
    #    We’ll place "white" (for 32-bit or 16-bit, that corresponds to a max value).
    #    Because base_image is mode "I", we’ll use an integer fill value—here, 65535 for 16-bit max.
    #    We want only the white letters to override the background.
    mask = scaled_text_img  # The non-zero pixels are the "letters" we want to paste

    # Create an Image in mode "I" filled with 65535 for the text region
    # This ensures the text region gets the maximum intensity (white) in 16-bit scale.
    text_layer = Image.new("I", (scaled_w, scaled_h), 65535)

    # Paste it into the base image at (x, y) using the scaled text image as a mask
    base_image.paste(text_layer, (x, y), mask)


def save_array_as_tiff(
    images: np.ndarray,
    file_path: str,
    min_val: float = None,
    max_val: float = None,
    divide_into_smaller_files: bool = True,
    numbers: np.ndarray = None,
    text_scale: float = 2.0,
):
    """
    Save a 3D NumPy array (images) to a TIFF file. Optionally, pass a 1D array (numbers)
    of the same length as images.shape[0], each of which is drawn in the top-left corner
    of each slice at a scaled-up size (using the built-in PIL font).

    :param images: 3D data array of shape (N, H, W).
    :param file_path: Path of the output tif file.
    :param min_val: Minimum value for scaling into uint16.
    :param max_val: Maximum value for scaling into uint16.
    :param divide_into_smaller_files: Whether to split the output into multiple smaller files
                                      if the total size exceeds ~4 GB.
    :param numbers: 1D array of length N, containing numeric values to overlay on each slice.
    :param text_scale: Factor by which to scale the default font size (>= 1.0).
    """
    # 1) Convert the input data to uint16
    images_uint16 = convert_to_uint_16(images, min_val, max_val)

    # 2) Overlay each slice with the corresponding number (if provided),
    #    using the built-in font and a scaling approach.
    if numbers is not None:
        if len(numbers) != images.shape[0]:
            raise ValueError("Length of 'numbers' must match the first dimension of 'images'.")

        # For each slice, convert to 32-bit ("I") for easy drawing in Pillow
        for i in range(images_uint16.shape[0]):
            # Convert to uint32 and then to a Pillow Image in mode 'I' (32-bit integer pixels)
            image_32 = images_uint16[i].astype(np.uint32)
            image_pil = Image.fromarray(image_32, mode="I")

            # Draw the text at (0, 0) using our upscaling function
            draw_text_with_default_font(
                base_image=image_pil,
                text=str(numbers[i]),
                x=0,
                y=0,
                scale_factor=text_scale,
            )

            # Convert back to uint16 and store in the original array
            images_uint16[i] = np.array(image_pil, dtype=np.uint16)

    # 3) Check if we need to split into smaller TIFF files for ImageJ compatibility
    if divide_into_smaller_files:
        max_file_size = 4 * 1e9  # ~4 GB limit
        if images_uint16.nbytes > max_file_size:
            n_files = int(np.ceil(images_uint16.nbytes / max_file_size))
            n_layers = images_uint16.shape[0]
            layers_per_file = int(np.ceil(n_layers / n_files))

            path, ext = os.path.splitext(file_path)
            for i in range(n_files):
                selected_layers = images_uint16[i * layers_per_file : (i + 1) * layers_per_file]
                selection_file_path = f"{path}_{i + 1}_of_{n_files}{ext}"
                tiff.imwrite(selection_file_path, selected_layers)
                print(f"File saved to: {selection_file_path}")
        else:
            tiff.imwrite(file_path, images_uint16)
            print(f"File saved to: {file_path}")
    else:
        tiff.imwrite(file_path, images_uint16)
        print(f"File saved to: {file_path}")
