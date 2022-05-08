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
from submission import generate_submission_file

from utils import get_nb_bands, get_scheduler, get_optimizer

from multitask_components import (
    DeepLabV2Decoder,
    DeeplabV2Encoder,
    BaseDecoder,
    MLPDecoder,
)
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
                orig_channels = self.encoder.conv1.in_channels
                weights = self.encoder.conv1.weight.data.clone()
                self.encoder.conv1 = nn.Conv2d(
                    get_nb_bands(self.bands),
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
                #assume first three channels are rgb

                if self.opts.module.pretrained:
                    self.encoder.conv1.weight.data[:, :orig_channels, :, :] = weights
                    
            self.avgpool = nn.Identity()
            self.encoder.fc = nn.Identity()  # nn.Linear(2048, self.target_size)

        if model == "resnet18":
            self.encoder = models.resnet18(pretrained=self.opts.module.pretrained)
            if get_nb_bands(self.bands) != 3:
                orig_channels = self.encoder.conv1.in_channels
                weights = self.encoder.conv1.weight.data.clone()
                self.ecoder.conv1 = nn.Conv2d(
                    get_nb_bands(self.bands),
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
                #assume first three channels are rgb

                if self.opts.module.pretrained:
                    self.encoder.conv1.weight.data[:, :orig_channels, :, :] = weights
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
            self.log(
                "val_country_loss",
                self.loss_country(out_country, meta["country"].unsqueeze(1)),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
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
            self.log(
                "val_country_loss",
                self.loss_country(out_country, meta["country"].unsqueeze(1)),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
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
      #  self.log(
     #       "val_country_loss",
    #        self.loss_country(out_country, meta["country"].unsqueeze(1)),
   #         on_step=False,
  #          on_epoch=True,
 #           sync_dist=True,
#        )
        #logging the metrics for val
         for (metric_name, _, scale) in self.metrics:
            nname = "val_" + metric_name
            metric_val = getattr(self, metric_name)(out_img.type_as(input_patches),  target)
            self.log(nname, metric_val, on_step = True, on_epoch = True)
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

        optimizer = get_optimizer(trainable_parameters, self.opts)
        scheduler = get_scheduler(optimizer, self.opts)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
