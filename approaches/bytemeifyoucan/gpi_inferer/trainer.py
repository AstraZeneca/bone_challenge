from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .config import Config
from .transforms import Crop, tf_aug, tf_crop, tf_postproc, tf_preproc, tf_window


def _get_inputs(batch_data: dict[str, Any], device: torch.device) -> torch.Tensor:
    return batch_data["img"].to(device)


def _get_targets(batch_data: dict[str, Any], device: torch.device) -> torch.Tensor:
    return batch_data["target"].to(device)


class Trainer:
    def __init__(
        self,
        config: Config,
        cache_rate: float = 0,
        num_workers: int = 1,
    ) -> None:
        self._config = config

        self._cache_rate = cache_rate
        self._num_workers = num_workers

        self._model = config.model(
            spatial_dims=2,
            in_channels=2 * config.inp_config.window,
            **asdict(config.model_config),
        ).to(config.device)

    @property
    def config(self) -> Config:
        return self._config

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def cache_rate(self) -> float:
        return self._cache_rate

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def run(
        self,
        fold: int,
        num_epochs: int = 1,
        val_interval: int = 1,
        save_interval: int = 1,
        checkpoint_path: Path | None = None,
        output_path: Path = Path(""),
    ) -> None:
        train_data, val_data = self.config.inp_config.get_data(fold)

        train_loader = DataLoader(
            dataset=CacheDataset(
                data=train_data,
                transform=Compose(
                    tf_preproc(self.config)
                    + tf_window(self.config)
                    + tf_aug(self.config)
                    + tf_crop(self.config, kind=Crop.RANDOM)
                    + tf_postproc()
                ),
                cache_rate=self.cache_rate,
            ),
            batch_size=self.config.inp_config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.config.pin_memory,
        )

        val_loader = (
            DataLoader(
                dataset=CacheDataset(
                    data=val_data,
                    transform=Compose(
                        tf_preproc(self.config)
                        + tf_window(self.config)
                        + tf_crop(self.config, kind=Crop.CENTER)
                        + tf_postproc()
                    ),
                    cache_rate=self.cache_rate,
                ),
                batch_size=self.config.inp_config.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.config.pin_memory,
            )
            if len(val_data) > 0
            else None
        )

        loss_function = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.train_config.lr
        )

        epoch = 0
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"] + 1

        pbar = tqdm(total=num_epochs, desc="epoch")
        pbar.update(epoch)
        writer = SummaryWriter()
        while epoch < num_epochs:
            self.model.train()
            epoch_loss = 0.0
            step = 0
            for batch_data in tqdm(train_loader, desc="training"):
                inputs = _get_inputs(batch_data, self.config.device)
                targets = _get_targets(batch_data, self.config.device)
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                step += 1

            epoch_loss /= step
            writer.add_scalar("train_loss", epoch_loss, epoch)

            if epoch % save_interval == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    output_path / f"model_{epoch:04d}.pth",
                )

            if epoch % val_interval == 0 and val_loader is not None:
                self.model.eval()
                epoch_loss = 0.0
                step = 0
                for batch_data in tqdm(val_loader, desc="validation"):
                    inputs = _get_inputs(batch_data, self.config.device)
                    targets = _get_targets(batch_data, self.config.device)
                    with torch.no_grad():
                        outputs = self.model(inputs).squeeze()
                        loss = loss_function(outputs, targets)
                        epoch_loss += loss.item()
                    step += 1

                epoch_loss /= step
                writer.add_scalar("val_loss", epoch_loss, epoch)

            epoch += 1
            pbar.update()
        writer.close()
