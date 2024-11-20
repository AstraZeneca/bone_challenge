from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose
from scipy.special import expit
from scipy.stats import norm
from tqdm.auto import tqdm

from .config import Config, get_data_from_dataframe
from .trainer import Trainer
from .transforms import Crop, tf_crop, tf_postproc, tf_preproc


def _load_model(
    model: torch.nn.Module, checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


class Predictor:
    def __init__(
        self,
        metadata: pd.DataFrame,
        images_path: Path,
        trainer: Trainer,
        cache_rate: float = 0,
        checkpoint_path: Path | None = None,
        device: torch.device | None = None,
        data_path: Path | None = None,
    ) -> None:
        self._trainer = trainer
        self._device = device or trainer.config.device

        self._model: torch.nn.Module | None = None
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)

        self._data = list(
            get_data_from_dataframe(
                metadata=metadata,
                images_path=images_path,
                data_path=data_path,
                num_slices=self.config.inp_config.dim_z,
            )
        )

        self._data_loader = DataLoader(
            dataset=CacheDataset(
                data=self.data,
                transform=Compose(
                    tf_preproc(self.config)
                    + tf_crop(self.config, kind=Crop.CENTER)
                    + tf_postproc()
                ),
                cache_rate=cache_rate,
            ),
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def config(self) -> Config:
        return self._trainer.config

    @property
    def data(self) -> list[dict[str, Any]]:
        return self._data

    @property
    def data_loader(self) -> DataLoader:
        return self._data_loader

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            raise ValueError("Please load a checkpoint before accessing the model.")
        return self._model

    def load_model(self, checkpoint_path: Path) -> None:
        self._model = _load_model(
            model=self._trainer.model,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )

    def run(
        self, batch_size: int = 1, data_path: Path | None = None
    ) -> dict[str, dict[str, Any]]:
        data = {}

        unfold_window, unfold_step = 2 * self.config.inp_config.window, 1
        for batch_data in tqdm(self.data_loader, desc="image"):
            name = str(batch_data["name"][0])
            try:
                gpi = int(batch_data["gpi"][0])
            except (KeyError, TypeError):
                # Non-training data will not have a GPI.
                gpi = None
            img = batch_data["img"][0]

            z = (
                torch.arange(self.config.inp_config.dim_z, dtype=torch.float32)  # type: ignore
                .unfold(0, unfold_window, unfold_step)
                .mean(axis=1)
                .numpy()
            )

            batch_inputs = (
                img.unfold(0, unfold_window, unfold_step)
                .moveaxis(-1, 1)
                .split(batch_size, dim=0)
            )

            with torch.no_grad():
                logits = np.vstack([
                    self.model(inputs.to(self.config.device)).cpu().detach().numpy()
                    for inputs in batch_inputs
                ])

            data[name] = {"z": z, "logits": logits, "gpi_true": gpi}

            if data_path is not None:
                with zarr.open(data_path, mode="a") as root:
                    group = root.create_group(name)
                    group.attrs["gpi_true"] = gpi
                    group.array("z", z)
                    group.array("logits", logits)
                    group.array("img", img.numpy())

        return data


class GPIInferer:
    def __init__(
        self,
        data: dict[str, Any] | None = None,
        data_path: Path | None = None,
        name: str | None = None,
        class_index: int = 0,
    ) -> None:
        self._name = name
        if data is not None:
            self._gpi_true = data["gpi_true"]
            self._z = data["z"]
            self._logits = data["logits"][:, class_index]
            self._img = None
        elif data_path is not None and name is not None:
            with zarr.open(data_path, mode="r") as root:
                group = root[name]
                self._gpi_true = group.attrs["gpi_true"]
                self._z = group["z"][:]
                self._logits = group["logits"][:, class_index]
                self._img = group["img"][:]
        else:
            raise ValueError("Need 'data' or both 'data_path' and 'name'.")

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def img(self) -> np.ndarray | None:
        return self._img

    @property
    def z(self) -> np.ndarray:
        return self._z

    @property
    def logits(self) -> np.ndarray:
        return self._logits

    @property
    def prob(self) -> np.ndarray:
        return expit(self.logits)

    @property
    def gpi_true(self) -> int:
        return self._gpi_true

    @property
    def gpi_pred(self) -> int:
        return round(self.z[np.argmax(self.prob)])

    @property
    def score(self) -> float | None:
        if self.gpi_true is None:
            return None
        return get_score(self.gpi_pred, self.gpi_true)

    def as_dict(self) -> dict[str, str | float | None]:
        data: dict[str, str | float | None] = {}
        if self.name is not None:
            data["name"] = self.name
        data["gpi_true"] = self.gpi_true
        data["gpi_pred"] = self.gpi_pred
        data["score"] = self.score
        return data


def get_score(gpi_pred: int, gpi_true: int, scale: int = 3) -> float:
    return 2 * norm.sf(abs(gpi_pred - gpi_true), 0, scale)


def get_num_model_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
