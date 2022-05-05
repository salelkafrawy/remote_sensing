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
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingWarmRestarts,
)
import pytorch_lightning as pl
import timm
from torchvision import models


from metrics.metrics_torch import predict_top_30_set
from metrics.metrics_dl import get_metrics

from multitask import DeepLabV2Decoder, DeeplabV2Encoder, BaseDecoder, MLPDecoder
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
    elif opts.scheduler.name == "CosineRestarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=opts.scheduler.cosine.t_0,
            T_mult=opts.scheduler.cosine.t_mult,
            eta_min=opts.scheduler.cosine.eta_min,
            last_epoch=opts.scheduler.cosine.last_epoch,
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
            meta[0].cpu().detach().numpy(),
            preds_30.cpu().detach().numpy(),
            append=True,
        )

        return output

    def get_optimizer(self, trainable_parameters, opts):

        if self.opts.optimizer == "Adam":
            optimizer = torch.optim.Adam(trainable_parameters, lr=self.learning_rate)
        elif self.opts.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(trainable_parameters, lr=self.learning_rate)
        elif self.opts.optimizer == "SGD":
            optimizer = torch.optim.SGD(trainable_parameters, lr=self.learning_rate)
        elif self.opts.optimizer == "SGD+Nesterov":
            optimizer = torch.optim.SGD(
                trainable_parameters,
                nesterov=self.nestrov,
                momentum=self.momentum,
                dampening=self.dampening,
                lr=self.learning_rate,
            )
        else:
            raise ValueError(f"Optimizer'{self.opts.optimizer}' is not valid")
        return optimizer

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

        optimizer = self.get_optimizer(trainable_parameters, self.opts)
        scheduler = get_scheduler(optimizer, self.opts)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class CNNMultitask(pl.LightningModule):
    def __init__(self, opts, **kwargs: Any) -> None:

        super().__init__()
        self.opts = opts
        self.bands = opts.data.bands
        self.target_size = opts.num_species
        self.learning_rate = self.opts.module.lr
        self.batch_size = self.opts.data.loaders.batch_size
        self.num_workers = self.opts.data.loaders.num_workers
        self.predict_country = self.opts.predict_country

        self.config_task(opts, **kwargs)

    def config_task(self, opts, **kwargs: Any) -> None:
        self.model_name = self.opts.module.model
        self.decoder_name = self.opts.module.decoder
        self.get_model(self.model_name)
        self.loss = nn.CrossEntropyLoss()
        self.loss_land = CrossEntropy()

        if self.predict_country:
            self.loss_country = BCE()

        metrics = get_metrics(self.opts)
        for (name, value, _) in metrics:
            setattr(self, name, value)
        self.metrics = metrics

    def get_model(self, model):
        if self.model_name == "deeplabv2":
            self.encoder = DeeplabV2Encoder(self.opts)
        elif self.model_name == "resnet50":
            self.encoder = models.resnet50(pretrained=self.opts.module.pretrained)

            if get_nb_bands(self.bands) != 3:
                self.encoder.conv1 = nn.Conv2d(
                    get_nb_bands(self.bands),
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
            self.avgpool = nn.Identity()
            self.encoder.fc = nn.Identity()  # nn.Linear(2048, self.target_size)

        if model == "resnet18":
            self.encoder = models.resnet18(pretrained=self.opts.module.pretrained)
            if get_nb_bands(self.bands) != 3:
                self.encoder.conv1 = nn.Conv2d(
                    get_nb_bands(self.bands),
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
            self.encoder.fc = nn.Linear(512, self.target_size)
        if self.decoder_name == "mlp":
            self.decoder_img = MLPDecoder(
                2048, self.target_size, flatten=(model == "deeplabv2")
            )

        elif self.decoder_name == "base":
            self.decoder_img = BaseDecoder(
                2048, self.target_size, flatten=(model == "deeplabv2")
            )

        if self.predict_country:
            self.decoder_country = BaseDecoder(2048, 1, flatten=(model == "deeplabv2"))

        self.decoder_land = DeepLabV2Decoder(self.opts)

    def forward(self, x: Tensor) -> Any:
        z = self.encoder(x)
        out_img = self.decoder_img(z)
        if self.model_name != "deeplabv2":
            out_land = self.decoder_land(z.unsqueeze(-1).unsqueeze(-1))
        else:
            out_land = self.decoder_land(z)
        if self.predict_country:
            out_country = self.decoder_country(z)
            return (out_img, out_land, out_country)
        else:
            return out_img, out_land

    def training_step(self, batch, batch_idx):
        patches, target, meta = batch
        longtensor = torch.zeros([1]).type(torch.LongTensor).cuda()
        input_patches = patches["input"]
        landcover = patches["landcover"]

        if self.predict_country:
            out_img, out_land, out_country = self.forward(input_patches)
            landcover = landcover.squeeze(1)
            loss = (
                self.loss(out_img, target)
                + self.loss_land(out_land, landcover)
                + self.loss_country(out_country, meta["country"].unsqueeze(1))
            )

        else:
            out_img, out_land = self.forward(input_patches)
            # out_img = out_img.type_as(target)
            landcover = landcover.squeeze(1)
            loss = self.loss(out_img, target) + self.loss_land(out_land, landcover)

        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        self.log(
            "img_loss",
            self.loss(out_img, target),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "land_loss",
            self.loss_land(out_land, landcover),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # logging the metrics for training
        # for (metric_name, _, scale) in self.metrics:
        #    nname = "train_" + metric_name
        #    metric_val = getattr(self, metric_name)(out_img.type_as(input_patches),  target)
        #    self.log(nname, metric_val, on_step = True, on_epoch = True)

        return loss

    def validation_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        patches, target, meta = batch
        input_patches = patches["input"]
        landcover = patches["landcover"]

        if self.predict_country:
            out_img, out_land, out_country = self.forward(input_patches)
            landcover = landcover.squeeze(1)
            loss = (
                self.loss(out_img, target)
                + self.loss_land(out_land, landcover)
                + self.loss_country(out_country, meta["country"].unsqueeze(1))
            )

        else:
            out_img, out_land = self.forward(input_patches)
            # out_img = out_img.type_as(target)
            landcover = landcover.squeeze(1)
            loss = self.loss(out_img, target) + self.loss_land(out_land, landcover)

        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(
            "val_img_loss",
            self.loss(out_img, target),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_land_loss",
            self.loss_land(out_land, landcover),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_country_loss",
            self.loss_country(out_country, meta["country"].unsqueeze(1)),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        patches, meta = batch
        input_patches = patches["input"]

        if self.predict_country:
            out_img, out_land, out_country = self.forward(input_patches)

        else:
            out_img, out_land = self.forward(input_patches)

        # generate submission file -> (36421, 30)
        probas = torch.nn.functional.softmax(out_img, dim=0)
        preds_30 = predict_top_30_set(probas)
        generate_submission_file(
            self.opts.preds_file,
            meta[0].cpu().detach().numpy(),
            preds_30.cpu().detach().numpy(),
            append=True,
        )

        return output

    def get_optimizer(self, trainable_parameters, opts):

        if self.opts.optimizer == "Adam":
            optimizer = torch.optim.Adam(trainable_parameters, lr=self.learning_rate)
        elif self.opts.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(trainable_parameters, lr=self.learning_rate)
        elif self.opts.optimizer == "SGD":
            optimizer = torch.optim.SGD(trainable_parameters, lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer'{self.opts.optimizer}' is not valid")
        return optimizer

    def configure_optimizers(self) -> Dict[str, Any]:

        parameters = (
            list(self.encoder.parameters())
            + list(self.decoder_img.parameters())
            + list(self.decoder_land.parameters())
        )
        if self.predict_country:
            parameters += list(self.decoder_country.parameters())

        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        print(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable components out of {len(parameters)}."
        )
        num_params = (
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            + sum(p.numel() for p in self.decoder_img.parameters() if p.requires_grad)
            + sum(p.numel() for p in self.decoder_land.parameters() if p.requires_grad)
        )
        print(
            f"Number of learnable parameters = {num_params} out of {len(parameters)} total parameters."
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
