import os
import sys
import inspect

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, CURR_DIR)
sys.path.insert(0, PARENT_DIR)

from re import L
import numpy as np
from PIL import Image
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module

import pytorch_lightning as pl
import timm
from torchvision import models


from metrics.metrics_torch import predict_top_30_set
from metrics.metrics_dl import get_metrics
from submission import generate_submission_file

from utils import get_nb_bands, get_scheduler, get_optimizer

from transformer import ViT


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, logits, target):
        return self.loss(logits, target.long())


class BCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, logits, target):
        return self.loss(logits, target.float())


class CNNBaseline(pl.LightningModule):
    def __init__(self, opts, **kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""

        super().__init__()
        self.opts = opts
        self.bands = opts.data.bands
        self.target_size = opts.num_species
        self.learning_rate = self.opts.module.lr
        self.nestrov = self.opts.nesterov
        self.momentum = self.opts.momentum
        self.dampening = self.opts.dampening
        self.batch_size = self.opts.data.loaders.batch_size
        self.num_workers = self.opts.data.loaders.num_workers
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
            if get_nb_bands(self.bands) != 3:
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
            if get_nb_bands(self.bands) != 3:
                self.model.conv1 = nn.Conv2d(
                    get_nb_bands(self.bands),
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
            self.model.fc = nn.Linear(2048, self.target_size)

        elif model == "inceptionv3":
            self.model = models.inception_v3(pretrained=self.opts.module.pretrained)
            self.model.AuxLogits.fc = nn.Linear(768, self.target_size)
            self.model.fc = nn.Linear(2048, self.target_size)

        elif model == "ViT":
            self.model = timm.create_model(
                "vit_base_patch16_224",
                pretrained=self.opts.module.pretrained,
                num_classes=self.target_size,
            )

        print(f"model inside get_model: {model}")

    def forward(self, x: Tensor) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.opts.use_ffcv_loader:
            rgb_arr, nearIR_arr, target = batch
            input_patches = rgb_arr
            if "near_ir" in self.bands:
                input_patches = torch.concatenate((rgb_arr, nearIR_arr), axis=0)

        else:
            patches, target, meta = batch
            input_patches = patches["input"]

        outputs = None
        if self.opts.module.model == "inceptionv3":
            outputs, aux_outputs = self.forward(input_patches)
            loss1 = self.loss(outputs, target)
            loss2 = self.loss(aux_outputs, target)
            loss = loss1 + loss2
        else:
            outputs = self.forward(input_patches)
            loss = self.loss(outputs, target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        # logging the metrics for training
        for (metric_name, _, scale) in self.metrics:
            nname = "train_" + metric_name
            metric_val = getattr(self, metric_name)(target, outputs)

            self.log(nname, metric_val, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        #import pdb; pdb.set_trace()
        if self.opts.use_ffcv_loader:
            rgb_arr, nearIR_arr, target = batch
            input_patches = rgb_arr
            if "near_ir" in self.bands:
                input_patches = torch.concatenate((rgb_arr, nearIR_arr), axis=0)

        else:
            patches, target, meta = batch
            input_patches = patches["input"]

        outputs = self.forward(input_patches)
        loss = self.loss(outputs, target)

        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        # logging the metrics for validation
        for (metric_name, _, scale) in self.metrics:
            nname = "val_" + metric_name
            metric_val = getattr(self, metric_name)(target, outputs)

            self.log(nname, metric_val, on_step=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        patches, meta = batch
        input_patches = patches["input"]

        output = self.forward(input_patches)
        
        # generate submission file -> (36421, 30)
        probas = torch.nn.functional.softmax(output, dim=0)
        preds_30 = predict_top_30_set(probas)
        generate_submission_file(
            self.opts.preds_file,
            meta['obs_id'].cpu().detach().numpy(),
            preds_30.cpu().detach().numpy(),
            append=True,
        )

        return output

    def configure_optimizers(self) -> Dict[str, Any]:

        parameters = list(self.model.parameters())

        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        print(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable components out of {len(parameters)}."
        )
        print(
            f"Number of learnable parameters = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} out of {sum(p.numel() for p in self.model.parameters())} total parameters."
        )

        optimizer = get_optimizer(trainable_parameters, self.opts)
        scheduler = get_scheduler(optimizer, self.opts)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
