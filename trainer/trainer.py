import os
import sys
import inspect

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, CURR_DIR)
sys.path.insert(0, PARENT_DIR)

from typing import Any, Dict, Optional
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
from metrics_torch import (
    top_30_error_rate,
    top_k_error_rate_from_sets,
    predict_top_30_set,
)
from submission import generate_submission_file
from torchvision import models
from metrics_dl import get_metrics


def get_nb_bands(bands):
    """
    Get number of channels in the satellite input branch
    (stack bands of satellite + environmental variables)
    """
    n = 0
    for b in bands:
        if b in ["near_ir", "landuse", "altitude"]:
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


class CNNBaseline(pl.LightningModule):
    def __init__(self, opts, **kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""

        super().__init__()
        self.opts = opts
        self.bands = opts.data.bands
        self.target_size = opts.num_species
        self.learning_rate = self.opts.module.lr
        self.config_task(opts, **kwargs)
        
    def config_task(self, opts, **kwargs: Any) -> None:
        self.model_name = self.opts.module.model
        self.get_model(self.model_name)
        self.loss = nn.CrossEntropyLoss()
        
        metrics = get_metrics(self.opts)
        for (name, value, _) in metrics:
            setattr(self, name, value)
        self.metrics = metrics

    def get_model(self, model):
        print(f"chosen model: {model}")
        if model == "resnet18":
            self.model = models.resnet18(pretrained=self.opts.module.pretrained)
            if len(self.opts.data.bands) != 3:
                self.model.conv1 = nn.Conv2d(
                    get_nb_bands(self.bands),
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
                    get_nb_bands(self.bands),
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
            self.model.fc = nn.Linear(512, self.target_size)

        elif model == "inceptionv3":
            self.model = models.inception_v3(pretrained=self.opts.module.pretrained)
            self.model.AuxLogits.fc = nn.Linear(768, self.target_size)
            self.model.fc = nn.Linear(2048, self.target_size)
        print(f"model inside get_model: {self.model}")

    def forward(self, x: Tensor) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        patches, target, meta = batch

        input_patches = patches['input']
        outputs = None
        if self.opts.module.model == "inceptionv3":
            outputs, aux_outputs = self.forward(input_patches)
            loss1 = self.loss(outputs, target)
            loss2 = self.loss(aux_outputs, target)
            loss = loss1 + loss2
        else:
            outputs = self.forward(input_patches)
            loss = self.loss(outputs, target)
        

        self.log("train_loss", loss, on_step = True, on_epoch= True)
        
        # logging the metrics for training
        for (metric_name, _, scale) in self.metrics:
            nname = "train_" + metric_name
            metric_val = getattr(self, metric_name)(target, outputs)
            
            self.log(nname, metric_val, on_step = True, on_epoch = True)
            
        return loss

    
    def validation_step(self, batch, batch_idx):
        patches, target, meta = batch

        input_patches = patches['input']
        outputs = self.forward(input_patches)
        loss = self.loss(outputs, target)

        self.log("val_loss", loss, on_step = True, on_epoch= True)
        # logging the metrics for validation
        for (metric_name, _, scale) in self.metrics:
            nname = "val_" + metric_name
            metric_val = getattr(self, metric_name)(target, outputs)
            
            self.log(nname, metric_val, on_step = True, on_epoch = True)


    def test_step(self, batch, batch_idx):
        patches, meta = batch
        input_patches = patches['input']
        output = self.forward(input_patches)
        return output

    def get_optimizer(self, trainable_parameters, opts):

        if self.opts.optimizer == "Adam":
            optimizer = torch.optim.Adam(  
                trainable_parameters, lr=self.learning_rate
            )
        elif self.opts.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                trainable_parameters, lr=self.learning_rate 
            )
        elif self.opts.optimizer == "SGD":
            optimizer = torch.optim.SGD(trainable_parameters, lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer'{self.opts.optimizer}' is not valid")
        return optimizer

    def configure_optimizers(self) -> Dict[str, Any]:

        parameters = list(self.model.parameters())

        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        print(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )

        optimizer = self.get_optimizer(trainable_parameters, self.opts)
        scheduler = get_scheduler(optimizer, self.opts)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
