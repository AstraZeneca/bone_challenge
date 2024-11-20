from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.networks import nets

DATA_PATH = Path("/scratch/BoneAI")
assert DATA_PATH.is_dir()


def get_data_from_dataframe(
    metadata: pd.DataFrame,
    images_path: Path,
    num_slices: int,
    data_path: Path | None = None,
) -> Iterator[dict[str, Any]]:
    for _, item in metadata.iterrows():
        name = item["Image Name"]
        try:
            gpi = item["Growth Plate Index"]
        except KeyError:
            # Non-training data will not have a GPI.
            gpi = -1

        # Load the NII image object.
        path = images_path / f"{name}.nii"
        image = nib.load(path)
        height, width, depth = image.shape  # type: ignore

        # Check that all images have the same number of slices.
        assert depth == num_slices

        # Check that all images have the same coordinate system.
        assert np.all(image.affine == np.identity(4))  # type: ignore
        assert np.all(image.header.get_zooms() == np.ones(3))  # type: ignore

        yield {
            "name": name,
            "img": path if data_path is None else name,
            "height": height,
            "width": width,
            "gpi": gpi,
            "seg": gpi,
        }


@dataclass(frozen=True)
class InputConfig:
    images_path: Path = DATA_PATH / "task2" / "images_3d"
    metadata_path: Path = DATA_PATH / "train_with_folds.csv"

    data_path: Path | None = None

    hu_range: tuple[int, int] = -100, 3171
    dim_xy_padded: int = 480  # 2 * dim_xy_resized
    dim_xy_resized: int = 240  # dim_xy + 16
    dim_xy: int = 224  # 7 * 2**
    dim_z: int = 642
    window: int = 32
    pos_odds: float = 1
    batch_size: int = 4
    num_samples: int = 8

    @property
    def metadata(self) -> pd.DataFrame:
        return pd.read_csv(self.metadata_path)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.dim_xy, self.dim_xy, self.dim_z

    @property
    def shape_resized(self) -> tuple[int, int, int]:
        return self.dim_xy_resized, self.dim_xy_resized, self.dim_z

    @property
    def shape_padded(self) -> tuple[int, int, int]:
        return self.dim_xy_padded, self.dim_xy_padded, self.dim_z

    def get_data(self, fold: int) -> tuple[list[dict[str, Any]], ...]:
        return tuple(
            list(
                get_data_from_dataframe(
                    metadata=metadata,
                    images_path=self.images_path,
                    data_path=self.data_path,
                    num_slices=self.dim_z,
                )
            )
            for metadata in (
                self.metadata[self.metadata["fold"] != fold],
                self.metadata[self.metadata["fold"] == fold],
            )
        )


@dataclass(frozen=True)
class AugmentationConfig:
    prob: float = 0.7
    angle_deg: float = 10


@dataclass(frozen=True)
class ModelConfig:
    out_channels: int = 1
    init_features: int = 64
    growth_rate: int = 32
    block_config: tuple[int, ...] = (4, 4, 4, 4, 4)
    bn_size: int = 4
    act: tuple[str, dict[str, Any]] = ("relu", {"inplace": True})
    norm: str = "batch"
    dropout_prob: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 1e-4


@dataclass(frozen=True)
class Config:
    inp_config: InputConfig = InputConfig()
    aug_config: AugmentationConfig = AugmentationConfig()
    model: type[torch.nn.Module] = nets.DenseNet
    model_config: ModelConfig = ModelConfig()
    train_config: TrainingConfig = TrainingConfig()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def pin_memory(self) -> bool:
        return torch.cuda.is_available()
