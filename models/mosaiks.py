import os
import sys
import inspect
from re import L
import numpy as np
from PIL import Image
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import Module
import pytorch_lightning as pl
from torchvision import models

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, CURR_DIR)
sys.path.insert(0, PARENT_DIR)

from metrics.metrics_torch import predict_top_30_set
from metrics.metrics_dl import get_metrics
from submission import generate_submission_file

from utils import get_nb_bands, get_scheduler, get_optimizer

from losses.PolyLoss import PolyLoss
import transforms.transforms as trf


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


class MOSAIKS(pl.LightningModule):
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
        
        self.pool_size = self.opts.mosaiks.pool_size
        self.pool_stride = self.opts.mosaiks.pool_stride
        self.patch_size = self.opts.mosaiks.patch_size
        self.in_channels = self.opts.mosaiks.in_channels
        self.bias = self.opts.mosaiks.bias
        self.num_feats = self.opts.mosaiks.num_feats
        self.patches_path = self.opts.mosaiks_weights_path
        
        self.patches_np = np.load(self.patches_path)
        
        self.config_task(opts, **kwargs)

    def config_task(self, opts, **kwargs: Any) -> None:
        self.model_name = self.opts.mosaiks.model_name
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
        
        if model == "one_layer":
            self.conv_layer = nn.Conv2d(in_channels=3, out_channels=100, kernel_size=7, padding='same', bias=True)
            self.conv_layer.load_state_dict(torch.load(self.opts.mosaiks_weights_path))
            self.conv_layer.weight.requires_grad = self.opts.mosaiks.finetune
            self.last_layer = nn.Linear(self.num_feats * 2, self.target_size)  
            self.model = nn.Sequential(self.conv_layer, self.last_layer)
            print(f'ONE layer model loaded from {self.opts.mosaiks_weights_path}')
            
        elif model == "two_layers":
            self.model = nn.Sequential(
                  nn.Conv2d(in_channels=3, out_channels=100, kernel_size=7, padding='same', bias=True),
                  nn.LeakyReLU(),
                  nn.MaxPool2d(2, stride=2),

                  nn.Conv2d(in_channels=100, out_channels=64, kernel_size=7, padding='same', bias=True),
                  nn.LeakyReLU(),
                  nn.MaxPool2d(2, stride=2),

                  nn.AdaptiveAvgPool2d(9),

                  nn.Flatten(),
                  nn.Dropout(0.5),
                  nn.Linear(5184, 512), #50176
                  nn.ReLU(),
                  nn.Linear(512, self.target_size)
                  ) 
            self.model.load_state_dict(torch.load(self.opts.mosaiks_weights_path))
            model[0].weight.requires_grad = self.opts.mosaiks.finetune
            model[3].weight.requires_grad = self.opts.mosaiks.finetune
            print(f'TWO layer model loaded from {self.opts.mosaiks_weights_path}')
        
        print(f"model inside get_model: {model}")

        
    def forward(self, x: Tensor) -> Any:
        
        if self.model_name == "one_layer":
            conv = self.conv_layer(x)

            x_pos = F.avg_pool2d(
                F.relu(conv - self.bias),
                [self.pool_size, self.pool_size],
                stride=[self.pool_stride, self.pool_stride],
                ceil_mode=True,
            )

            x_neg = F.avg_pool2d(
                F.relu((-1 * conv) - self.bias),
                [self.pool_size, self.pool_size],
                stride=[self.pool_stride, self.pool_stride],
                ceil_mode=True,
            )

            cat_vec = torch.cat((x_pos, x_neg), dim=1)
            cat_vec = cat_vec.view(cat_vec.size(0), -1)

            return self.last_layer(cat_vec)
        else:
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
