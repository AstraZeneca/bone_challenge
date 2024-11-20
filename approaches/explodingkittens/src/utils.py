# Import libraries
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import nibabel as nib
import numpy as np
import cv2
from glob import glob

LABEL_REGEX = r"gp_([\d]{3})"


def show_slices(slices, plate_idx=None):
    """Function to display row of image slices"""
    _, axes = plt.subplots(1, len(slices), figsize=(10, 7))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="bone", origin="lower")

        if plate_idx is not None:
            if i != len(slices) - 1:
                axes[i].hlines(plate_idx, xmin=0, xmax=slice.shape[0] - 1, color="r")


def get_image_info(row):
    path = row.image_path
    img = nib.load(path)
    x, y, z = img.shape

    row["slices_per_dimension"] = (x, y, z)
    row["voxel_size"] = img.header.get_zooms()
    row["measurement_units"] = img.header.get_xyzt_units()

    row["x_size"] = x
    row["y_size"] = y
    row["z_size"] = z

    return row


def get_metadata(metadata_path: str, images_path: str):
    metadata_df = pd.read_csv(metadata_path)

    metadata_df["image_path"] = metadata_df["Image Name"].map(
        lambda x: Path(f"{images_path}/{x}.nii")
    )
    metadata_df = metadata_df.apply(get_image_info, axis=1)

    return metadata_df


def apply_bone_window(image, window_width, window_level, bit_depth=16):

    lower_limit = window_level - (window_width / 2)
    upper_limit = window_level + (window_width / 2)

    # Apply the window settings
    windowed_image = np.clip(image, lower_limit, upper_limit)

    return windowed_image


def equalize_hist_16bit(image):
    # Flatten the image to get the image histogram
    hist, bins = np.histogram(image.flatten(), 65536, [0, 65536])

    # Cumulative distribution function
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 65535 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint16)

    # Use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)

    return image_equalized.reshape(image.shape)


def normalize_image(image, bit_depth=16):
    min_v, max_v = image.min(), image.max()
    # Normalize the windowed image to the range 0-255 if needed (for display purposes)
    image = ((image - min_v) / (max_v - min_v)) * (2**bit_depth - 1)
    image = image.astype(np.uint8) if bit_depth == 8 else image.astype(np.uint16)

    return image


def pad_crop_image(img, size):
    h, w = img.shape

    if h > size[0]:
        # Resize the image to the desired width
        img = img[: size[0], :]
        # Resize the mask if provided with interpolation set to nearest to keep pixel values

    if w > size[1]:
        img = img[:, : size[1]]
    # If the width of the image is less than the desired width

    if w < size[1]:
        # Add padding to the right side of the image to reach the desired width
        img = cv2.copyMakeBorder(
            img, 0, 0, 0, size[0] - w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    return img


def load_nii_file(file_path):
    return nib.load(file_path).get_fdata()


def get_all_images(image_dir: str, ext="png"):
    return glob(f"{image_dir}/*.{ext}")


def calculate_score(pred_slice_num, gt_slice_num):
    """Returns the survival function a single-sided normal distribution with stddev=3."""
    diff = abs(pred_slice_num - gt_slice_num)
    return 2 * norm.sf(diff, 0, 3)
