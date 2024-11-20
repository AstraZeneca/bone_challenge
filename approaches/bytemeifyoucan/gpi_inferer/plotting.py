from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .config import Config
from .math import moving_average
from .predictor import GPIInferer


class Plotter:
    def __init__(self, config: Config) -> None:
        self.config = config

    def plot_loss(self, run: Path, ax: Any = None, window: int = 0, **kwargs) -> Any:
        if ax is None:
            _, ax = plt.subplots(**kwargs)

        acc = EventAccumulator(str(run)).Reload()

        labels = {"train_loss": "Training", "val_loss": "Validation"}
        for scalar, color in [("train_loss", "C0"), ("val_loss", "C1")]:
            try:
                step, value = np.transpose([
                    (record.step, record.value) for record in acc.Scalars(scalar)
                ])
            except KeyError:
                continue

            label = labels.get(scalar)
            smooth = partial(moving_average, window=window)

            if window > 1:
                marker_style = dict(marker=".", mec=color, mfc="none", alpha=0.5)
                ax.plot(step, value, ls="none", **marker_style)

            step, value = map(smooth, (step, value))
            ax.plot(step, value, color=color, ls="-", label=label)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        ax.legend()

        return ax

    def plot_prediction(self, inf: GPIInferer, ax: Any = None, **kwargs) -> Any:
        if ax is None:
            _, ax = plt.subplot_mosaic(
                [["prob"], ["prob_zoom"], ["img_top"], ["img_bottom"]],
                height_ratios=[1, 0.5, 1, 1],
                sharex=True,
                **kwargs,
            )

        gpi_true, gpi_pred, img = inf.gpi_true, inf.gpi_pred, inf.img
        assert img is not None

        window = self.config.inp_config.window
        smooth = partial(moving_average, window=window)

        for i in ["prob", "prob_zoom", "img_top", "img_bottom"]:
            if gpi_true >= 0:
                ax[i].axvline(gpi_true, color="red", ls="-")
            ax[i].axvline(gpi_pred, color="red", ls="--")

        for i in ["prob", "prob_zoom"]:
            ax[i].plot(inf.z, inf.prob, color="k")
            ax[i].plot(smooth(inf.z), smooth(inf.prob), color="k", ls=":")
            ax[i].set_ylabel("Probability")

        ax["prob"].set_ylim(0, 1)
        ax["prob_zoom"].set_ylim(np.max(inf.prob) - 0.1, np.max(inf.prob) + 0.05)

        ax["img_top"].imshow(img[:, img.shape[1] // 2].T, vmin=0, vmax=1)
        ax["img_bottom"].imshow(img[:, :, img.shape[2] // 2].T, vmin=0, vmax=1)

        for i in ["img_top", "img_bottom"]:
            if gpi_true >= 0:
                for offset in [-window, window]:
                    ax[i].axvline(gpi_true + offset, color="orange", ls=":")
            ax[i].set_aspect("auto")

        ax["img_top"].set_ylabel("Width")
        ax["img_bottom"].set_ylabel("Height")
        ax["img_bottom"].set_xlabel("Depth (Z)")

        title = (
            f"Score: {inf.score:.3f} (Predicted: {gpi_pred}, Actual: {gpi_true})"
            if gpi_true >= 0
            else f"Predicted GPI: {gpi_pred}"
        )
        ax["prob"].set_title(title)

        return ax
