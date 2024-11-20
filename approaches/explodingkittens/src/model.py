import lightning.pytorch as pl
import timm
import torch
from torch import nn
from torchmetrics import Metric
from scipy.stats import norm


class SurvivalFunctionScore(Metric):
    def __init__(self, dist_stddev=3, **kwargs):
        """
        Initialize the SurvivalFunctionScore metric.

        Args:
        dist_stddev (float): The standard deviation of the normal distribution used in the survival function.
        """
        super().__init__(**kwargs)
        self.dist_stddev = dist_stddev
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        """
        Update state with predictions and true values.

        Args:
        preds (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        """
        preds, target = preds.detach().cpu(), target.detach().cpu()
        diff = torch.abs(preds - target)
        score = 2 * torch.tensor(norm.sf(diff, 0, self.dist_stddev))
        self.score_sum += torch.sum(score).to(self.score_sum.device)
        self.total += target.numel()

    def compute(self):
        """
        Compute the final score.
        """
        return self.score_sum / self.total


class BoneNet(pl.LightningModule):
    def __init__(self, model_name, lr=1e-3, wd=1e-3):
        super().__init__()

        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr
        self.wd = wd
        self.model = timm.create_model(self.model_name, pretrained=True, in_chans=3)
        in_features = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, out_features=1), nn.Sigmoid()
        )

        self.loss_fn = nn.MSELoss()
        self.val_sf_score = SurvivalFunctionScore()
        self.final_score = 0

    def forward(self, img):
        out = self.model(img)
        return out.squeeze()

    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self._shared_step(batch)
        self.log("valid_loss", loss, prog_bar=True)

        self.val_sf_score(y_hat, y)
        self.log(
            "survival_score_val",
            self.val_sf_score,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

    def on_validation_epoch_end(self):
        # At the end of each validation epoch
        val_score = self.val_sf_score.compute()
        self.final_score = val_score.item()

    def predict(self, x):
        y_hat = self(x.unsqueeze(0)) * x.shape[1]
        return y_hat

    def _compute_loss(self, y_hat, y):
        return torch.sqrt(self.loss_fn(y_hat, y))

    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self._compute_loss(y_hat, y)
        y, y_hat = y * x.shape[2], y_hat * x.shape[2]

        return loss, y, y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
