import os
import sys
import inspect
from re import L
import numpy as np
from PIL import Image
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
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


def get_first_layer_weights(opts, bands, net):
    # for resnets, vgg16 would be net.features[0] instead of net.conv1
    orig_channels = net.conv1.in_channels
    weights = net.conv1.weight.data.clone()
    net.conv1 = nn.Conv2d(
        get_nb_bands(bands),
        64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    )
    # assume first three channels are rgb
    if opts.module.pretrained:
        net.conv1.weight.data[:, :orig_channels, :, :] = weights


def load_moco_weights(ckpt_path, net, opts):
    if os.path.isfile(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)

        # rename moco pre-trained keys
        state_dict = checkpoint["state_dict"]

        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            # for SSL4EO
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # for SEN12MS
            elif k.startswith("backbone2") and not k.startswith("backbone2.fc"):
                state_dict[k[len("backbone2.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        """
        # remove prefix
        state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}
        """

        # get the RGB bands only
        # all bands: ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        # RGB bands = ['B04', 'B03', 'B02']
        # NearIR = B08
        tmp = state_dict["conv1.weight"][:, 1:4, :, :]
        tmp = torch.flip(tmp, dims=[1])

        if get_nb_bands(opts.data.bands) == 4:
            nearir = state_dict["conv1.weight"][:, 7, :, :].unsqueeze(1)
            tmp = torch.cat((tmp, nearir), axis=1)

        state_dict["conv1.weight"] = tmp
        msg = net.load_state_dict(state_dict, strict=False)

        # pdb.set_trace()
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(ckpt_path))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_path))

        
def zero_aware_normalize(embedding, axis):
    normalized = torch.nn.functional.normalize(embedding, p=2, dim=axis)
    norms = torch.norm(embedding, p=2, dim=axis, keepdim=True)
    is_zero_norm = (norms == 0).expand_as(normalized)
    return torch.where(is_zero_norm, torch.zeros_like(embedding), normalized)


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

        if model == "vgg16":
            self.model = models.vgg16(pretrained=self.opts.module.pretrained)
            if get_nb_bands(self.bands) != 3:
                get_first_layer_weights(self.opts, self.bands, self.model)

            fc_layer = nn.Linear(25088, self.target_size)
            self.model = nn.Sequential(
                self.model.features, self.model.avgpool, nn.Flatten(), fc_layer
            )

        if model == "resnet18":
            self.model = models.resnet18(pretrained=self.opts.module.pretrained)
            if get_nb_bands(self.bands) != 3:
                get_first_layer_weights(self.opts, self.bands, self.model)
            self.model.fc = nn.Linear(512, self.target_size)

        elif model == "resnet50":

            self.model = models.resnet50(pretrained=self.opts.module.pretrained)
            
            if self.opts.module.is_head2toe:
                self.model.fc = nn.Linear(2048+1024+512+256, self.target_size)
                self.model.fc.weight.requires_grad = True
                self.model.fc.bias.requires_grad = True
            else: 
                self.model.fc = nn.Linear(2048, self.target_size)
                
            if get_nb_bands(self.bands) != 3:
                get_first_layer_weights(self.opts, self.bands, self.model)

            if self.opts.module.custom_init:
                print("CUSTOM INIT IS LOADING ...")
                if self.opts.module.submodel == "mocov2_encoder":
                    load_moco_weights(self.opts.cnn_ckpt_path, self.model, self.opts)
                elif (
                    self.opts.module.submodel == "original"
                ):  # straight forward resnet50 model
                    self.model.load_state_dict(torch.load(self.opts.cnn_ckpt_path))
                print(f"Custom resnet50 loaded from {self.opts.cnn_ckpt_path}")

            if self.opts.module.freeze:
                # freeze the resnet's parameters
                num_layers = len(list(self.model.children()))
                count = 0
                for child in self.model.children():
                    if count < num_layers - 1:
                        for param in child.parameters():
                            param.requires_grad = False
                    count += 1

        elif model == "inceptionv3":
            self.model = models.inception_v3(pretrained=self.opts.module.pretrained)
            self.model.AuxLogits.fc = nn.Linear(768, self.target_size)
            self.model.fc = nn.Linear(2048, self.target_size)

        print(f"model inside get_model: {model}")

    def forward(self, x: Tensor) -> Any:
        if self.opts.module.is_head2toe:
            out = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
 
            out1 = self.model.layer1(out)
            out2 = self.model.layer2(out1)
            out3 = self.model.layer3(out2)
            out4 = self.model.layer4(out3)
            
            feat1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))(out1).squeeze(-1).squeeze(-1)
            feat2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))(out2).squeeze(-1).squeeze(-1)
            feat3 = nn.AdaptiveAvgPool2d(output_size=(1, 1))(out3).squeeze(-1).squeeze(-1)
            feat4 = nn.AdaptiveAvgPool2d(output_size=(1, 1))(out4).squeeze(-1).squeeze(-1)
            feats = torch.cat((feat1, feat2, feat3, feat4), axis = 1)
            
            # normalize feats
            # feats_each = [zero_aware_normalize(e, axis=1) for e in feats]
            # embeddings = tf.concat(feats_each, -1)
            feats_normalized = zero_aware_normalize(feats, axis=1)
            return self.model.fc(feats_normalized)
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
