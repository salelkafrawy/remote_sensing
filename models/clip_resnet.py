import os
import sys
import inspect
from re import L
import numpy as np
from PIL import Image
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingWarmRestarts,
)
import pytorch_lightning as pl
import clip


CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, CURR_DIR)
sys.path.insert(0, PARENT_DIR)

from utils import get_nb_bands, get_scheduler, get_optimizer, zero_aware_normalize
from losses.PolyLoss import PolyLoss
import transforms.transforms as trf
from metrics.metrics_torch import predict_top_30_set
from metrics.metrics_dl import get_metrics
from submission import generate_submission_file


 
class CLIPResnetCNN(pl.LightningModule):
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
        if self.opts.loss == "CrossEntropy":
            self.loss = nn.CrossEntropyLoss()
        elif self.opts.loss == "PolyLoss":
            self.loss = PolyLoss(softmax=True)

        metrics = get_metrics(self.opts)
        for (name, value, _) in metrics:
            setattr(self, name, value)
        self.metrics = metrics

    def get_model(self, model):
        print(f"chosen model: {model}")

        if self.opts.testing:
            # NOT IMPLEMENTED
            print('not implemented')
#             seco_model = MocoV2(self.opts)
#             resnet_model = deepcopy(seco_model.encoder_q)
#             if get_nb_bands(self.bands) != 3:
#                 get_first_layer_weights(self.opts, self.bands, resnet_model)
#             fc_layer = nn.Linear(2048, self.target_size)
#             self.model = nn.Sequential(resnet_model, fc_layer)
#             load_encoder_for_testing(self.opts.cnn_ckpt_path, self.model)

        else:
            clip_model, preprocess = clip.load("RN50", device='cpu')
            clip_model.eval()
            resnet_clip = clip_model.visual

            if get_nb_bands(self.bands) != 3:
                orig_channels = resnet_clip.conv1.in_channels
                weights = resnet_clip.conv1.weight.data.clone()
                resnet_clip.conv1 = nn.Conv2d(
                    get_nb_bands(self.bands),
                    32,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                )
                # assume first three channels are rgb
                resnet_clip.conv1.weight.data[:, :orig_channels, :, :] = weights

            fc_layer = nn.Linear(1024, self.target_size)
            self.model = nn.Sequential(resnet_clip, fc_layer)

        if self.opts.module.freeze:
            # NEED TO CHECK THE LOGIC WORKS HERE
            # freeze the resnet's parameters (except last layer)
            num_layers = len(list(self.model.children()))
            count = 0
            for child in self.model.children():
                if count < num_layers - 1:
                    for param in child.parameters():
                        param.requires_grad = False
                count += 1

        print(f"model inside get_model: {model}")
        print(f"Custom CLIP resnet50 loaded from {self.opts.cnn_ckpt_path}")

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.opts.use_ffcv_loader:
            rgb_patches, target = batch
            patches = {}
            patches["input"] = rgb_patches
            return patches, target
        else:
            if self.trainer.training:
                patches, target, meta = batch
                patches = trf.get_transforms(self.opts, "train")(
                    patches
                )  # => we perform GPU/Batched data augmentation
                if self.opts.task == "multimodal":
                    patches["env_vars"] = patches["env_vars"].type(torch.float16)
            elif self.trainer.testing:
                patches, meta = batch
                patches = trf.get_transforms(self.opts, "val")(patches)
                if self.opts.task == "multimodal":
                    patches["env_vars"] = patches["env_vars"].type(torch.float32)
            else:
                patches, target, meta = batch
                patches = trf.get_transforms(self.opts, "val")(patches)
                if self.opts.task == "multimodal":
                    patches["env_vars"] = patches["env_vars"].type(torch.float16)

            first_band = self.bands[0]
            patches["input"] = patches[first_band]

            for idx in range(1, len(self.bands)):
                patches["input"] = torch.cat(
                    (patches["input"], patches[self.bands[idx]]), axis=1
                )

            if self.trainer.testing:
                return patches, meta
            else:
                return patches, target, meta

    def training_step(self, batch, batch_idx):
        if self.opts.use_ffcv_loader:
            patches, target = batch
        else:
            patches, target, meta = batch
        input_patches = patches["input"]

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
            patches, target = batch

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
        probas = torch.nn.functional.softmax(output, dim=1)
        preds_30 = predict_top_30_set(probas)
        generate_submission_file(
            self.opts.preds_file,
            meta["obs_id"].cpu().detach().numpy(),
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
        scheduler = get_scheduler(
            optimizer, self.opts, len(self.trainer.datamodule.train_dataset)
        )

        if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
            interval = "epoch"
        else:
            interval = "step"
        lr_scheduler = {
            "scheduler": scheduler,
            "interval": interval,
            "monitor": "val_loss",
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
