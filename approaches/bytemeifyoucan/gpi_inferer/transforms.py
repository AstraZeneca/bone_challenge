from enum import Enum
from functools import partial

import numpy as np
import zarr
from monai.transforms import (
    CenterSpatialCropd,
    DeleteItemsd,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    MapTransform,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandSpatialCropd,
    Resized,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    SqueezeDimd,
)

from .config import Config

Keys = list[str]

LAZY = True


class Crop(Enum):
    RANDOM = "random"
    CENTER = "center"


def _img(name: str, config: Config) -> np.ndarray:
    data_path = config.inp_config.data_path
    assert data_path is not None

    with zarr.open(data_path, mode="r") as group:
        return group[name][:]


def _seg(gpi: int, config: Config) -> np.ndarray:
    shape = config.inp_config.shape_resized
    window = config.inp_config.window

    data = np.zeros(shape=(1, *shape), dtype=np.float32)
    if gpi < 0:
        return data

    for i in range(window):
        value = (window - i) / window
        data[..., gpi - i] = value
        data[..., gpi + i] = value

    return data


def tf_preproc(config: Config) -> list[MapTransform]:
    hu_range = config.inp_config.hu_range
    shape_padded = config.inp_config.shape_padded
    shape_resized = config.inp_config.shape_resized

    return [
        # Load the image data.
        (
            LoadImaged("img", reader="NibabelReader", image_only=True)
            if config.inp_config.data_path is None
            else Lambdad("img", partial(_img, config=config))
        ),
        # Add a single channel dimension to be able to use 3D spatial transforms.
        EnsureChannelFirstd("img", channel_dim="no_channel"),
        # Rescale the HU values.
        ScaleIntensityRanged("img", *hu_range, 0.0, 1.0, clip=True),
        # Resize to common initial dimensions.
        ResizeWithPadOrCropd("img", shape_padded, mode="constant", value=0, lazy=LAZY),
        # Resize to the target dimensions.
        Resized("img", shape_resized, mode="area", lazy=LAZY),
        # Mark all potential window centers.
        Lambdad("seg", partial(_seg, config=config)),
    ]


def tf_window(config: Config) -> list[MapTransform]:
    return [
        # Extract a random window.
        RandCropByPosNegLabeld(
            ["img", "seg"],
            label_key="seg",
            spatial_size=[-1, -1, 2 * config.inp_config.window],
            pos=config.inp_config.pos_odds,
            neg=1,
            num_samples=config.inp_config.num_samples,
            lazy=LAZY,
        ),
        # Assign target values for each window, according to the distance to the GP.
        Lambdad(
            "seg",
            lambda seg: seg[0, seg.shape[1] // 2, seg.shape[2] // 2, seg.shape[3] // 2],
            overwrite="target",
        ),
        # Delete the obsolete mask.
        DeleteItemsd("seg"),
    ]


def tf_aug(config: Config) -> list[MapTransform]:
    prob = config.aug_config.prob
    angle_rad = config.aug_config.angle_deg * np.pi / 180

    return [
        # Apply geometric augmentations without resampling.
        RandFlipd("img", prob=prob, spatial_axis=(0, 1), lazy=LAZY),
        RandRotate90d("img", prob=prob, spatial_axes=(0, 1), lazy=LAZY),
        # Apply geometric augmentations with resampling.
        RandRotated(
            "img",
            prob=prob,
            range_z=angle_rad,
            mode="bilinear",
            padding_mode="zeros",
            lazy=LAZY,
        ),
    ]


def tf_crop(
    config: Config, keys: Keys = ["img"], kind: Crop = Crop.RANDOM
) -> list[MapTransform]:
    cropper = CenterSpatialCropd if kind == Crop.CENTER else RandSpatialCropd

    roi_size = list(config.inp_config.shape)
    roi_size[2] = -1

    return [
        # Perform a crop to the target dimensions.
        cropper(keys, roi_size=roi_size, lazy=LAZY),
    ]


def tf_postproc(keys: Keys = ["img"]) -> list[MapTransform]:
    return [
        # Remove the previously added, single channel dimension.
        SqueezeDimd(keys),
        # Convert to channels-first format.
        EnsureChannelFirstd(keys, channel_dim=2),
    ]
