from re import L
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingWarmRestarts,
)
from torch import Tensor
from torch.nn.modules import Module
from GLC.metrics import (
    top_30_error_rate,
    top_k_error_rate_from_sets,
    predict_top_30_set,
)
from GLC.submission import generate_submission_file
from torchvision import models


def get_nb_bands(bands):
    """
    Get number of channels in the satellite input branch (stack bands of satellite + environmental variables)
    """
    n = 0
    for b in bands:
        if b in ["nir", "landuse", "altitude"]:
            n += 1
        elif b == "ped":
            n += 8
        elif b == "bioclim":
            n += 19
        elif b == "rgb":
            n += 3
    return n


def get_scheduler(optimizer, opts):
    if opts.scheduler.name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            factor=opts.scheduler.reduce_lr_plateau.factor,
            patience=opts.scheduler.reduce_lr_plateau.lr_schedule_patience,
        )
    elif opts.scheduler.name == "StepLR":
        return StepLR(
            optimizer, opts.scheduler.step_lr.step_size, opts.scheduler.step_lr.gamma
        )
    elif opts.scheduler.name is None:
        return None

    else:
        raise ValueError(f"Scheduler'{opts.scheduler.name}' is not valid")


class GLCTask(pl.LightningModule):
    def __init__(self, opts, **kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""

        super().__init__()
        self.opts = opts
        self.bands = opts.bands
        self.target_size = opts.num_species

    def config_task(self, **kwargs: Any) -> None:
        self.model_name = self.opts.experiment.module.model
        self.model = self.get_model(self.model_name)
        self.criterion = nn.CrossEntropyLoss()

    def get_model(self, model):
        if model == "resnet18":
            self.model = models.resnet18(pretrained=self.opts.module.pretrained)
            if len(self.opts.data.bands) != 3:
                self.model.conv1 = nn.Conv2d(
                    get_nb_bands(bands),
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
            self.model.fc = nn.Linear(512, self.target_size)

        elif model == "resnet50":
            self.model = models.resnet50(pretrained=self.opts.module.pretrained)
            if len(self.opts.data.bands) != 3:
                self.model.conv1 = nn.Conv2d(
                    get_nb_bands(bands),
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
            self.model.fc = nn.Linear(512, self.target_size)

        elif model == "inceptionv3":
            self.model = models.inception_v3(
                pretrained=self.opts.experiment.module.pretrained
            )
            self.model.AuxLogits.fc = nn.Linear(768, self.target_size)
            self.model.fc = nn.Linear(2048, self.target_size)

    def forward(self, x: Tensor) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        patches, target, meta = batch

        if self.opts.experiment.module.model == "inceptionv3":
            y_hat, aux_outputs = self.forward(patches)
            loss1 = self.loss(y_hat, target)
            loss2 = self.loss(aux_outputs, target)
            loss = loss1 + loss2
        else:
            input = self.forward(patches)
            loss = self.criterion(input, target)
        return loss

    def validation_step(self, batch, batch_idx):
        patches, target, meta = batch
        input = self.forward(patches)
        loss = self.criterion(input, target)
        # acc = accuracy(y_hat, y)
        metrics = {
            "loss": loss,
        }
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        patches, meta = batch
        input = self.forward(patches)
        return input

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.SGD(  # Adam(   #
            self.model.parameters(),
            lr=self.opts.experiment.module.lr,  # CHECK IN CONFIG
        )

        scheduler = get_scheduler(optimizer, self.opts)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss",},
        }
