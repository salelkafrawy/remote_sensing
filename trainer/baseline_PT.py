import os
import sys
import inspect

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, CURR_DIR)
sys.path.insert(0, PARENT_DIR)

from typing import Any, Dict, Optional
from re import L
from tqdm import tqdm

# import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingWarmRestarts,
)
from torch import Tensor
from torch.nn.modules import Module
from metrics.metrics_torch import predict_top_30_set
from submission import generate_submission_file
from torchvision import models
from metrics.metrics_dl import get_metrics

from torchvision import transforms
from multitask import DeepLabV2Decoder, DeeplabV2Encoder, BaseDecoder
from transformer import ViT
from torch.utils.data import DataLoader
from data_loading.pytorch_dataset import GeoLifeCLEF2022Dataset
import transforms.transforms as trf

import numpy as np
from PIL import Image
from data_loading.ffcv_loader.dataset_ffcv import GeoLifeCLEF2022DatasetFFCV
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, NDArrayField
from ffcv.fields.decoders import (
    IntDecoder,
    NDArrayDecoder,
    SimpleRGBImageDecoder,
    CenterCropRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    RandomHorizontalFlip,
    Cutout,
    NormalizeImage,
    RandomTranslate,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
    ImageMixup,
)


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, logits, target):
        return self.loss(logits, target.long())


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


class CNNBaselinePT(nn.Module):
    def __init__(self, opts, device, **kwargs: Any) -> None:
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
        self.device = device

    def prepare_training(self) -> None:
        self.val_loader = self.val_dataloader()
        self.train_loader = self.train_dataloader()
        opt_configs = self.configure_optimizers()
        self.optimizer = opt_configs["optimizer"]
        self.lr_scheduler = opt_configs["lr_scheduler"]
        self.model.to(self.device)

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
            self.model = ViT(
                image_size=224,
                patch_size=32,
                num_classes=self.target_size,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                pool="cls",
                channels=3,
                dim_head=64,
                dropout=0.1,
                emb_dropout=0.1,
            )

        print(f"model inside get_model: {model}")

    def forward(self, x: Tensor) -> Any:
        return self.model(x)

    def train_dataloader(self):
        if self.opts.use_ffcv_loader:
            train_dataset = GeoLifeCLEF2022DatasetFFCV(
                self.opts.dataset_path,
                self.opts.data.splits.train,  # "train+val"
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
            )

            write_path = os.path.join(self.opts.save_path, "geolife_train_data.beton")
            # Pass a type for each data field
            writer = DatasetWriter(
                write_path,
                {
                    # Tune options to optimize dataset size, throughput at train-time
                    "rgb": RGBImageField(max_resolution=256),
                    "near_ir": NDArrayField(
                        dtype=np.dtype("float32"), shape=(1, 256, 256)
                    ),
                    "label": IntField(),
                },
            )

            # Write dataset
            writer.from_indexed_dataset(train_dataset)

            # Data decoding and augmentation (the first one is the left-most)
            img_pipeline = [
                CenterCropRGBImageDecoder(output_size=(224, 224), ratio=0.5),
                RandomHorizontalFlip(flip_prob=0.5),
                ImageMixup(alpha=0.5, same_lambda=True),
                ToTensor(),
                #                 ToDevice(self.device, non_blocking=True),
                ToTorchImage(),
                NormalizeImage(
                    np.array([106.9413, 114.8729, 104.5280]),
                    np.array([51.0005, 44.8594, 43.2014]),
                    np.float16,
                ),
            ]

            input_pipeline = [
                NDArrayDecoder(),
                ToTensor(),
                #                 ToDevice(self.device, non_blocking=True),
                transforms.Normalize([131.0458], [53.0884]),
            ]

            label_pipeline = [
                IntDecoder(),
                ToTensor(),
            ]
            #                 ToDevice(self.device, non_blocking=True)]

            # Pipeline for each data field
            pipelines = {
                "rgb": img_pipeline,
                "near_ir": input_pipeline,
                "label": label_pipeline,
            }

            train_loader = Loader(
                write_path,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                order=OrderOption.RANDOM,
                pipelines=pipelines,
            )

        else:

            # data and transforms
            train_dataset = GeoLifeCLEF2022Dataset(
                self.opts.dataset_path,
                self.opts.data.splits.train,  # "train+val"
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=trf.get_transforms(
                    self.opts, "train"
                ),  # transforms.ToTensor(),
                target_transform=None,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                #             pin_memory=True,
            )
        return train_loader

    def val_dataloader(self):
        if self.opts.use_ffcv_loader:
            val_dataset = GeoLifeCLEF2022DatasetFFCV(
                self.opts.dataset_path,
                self.opts.data.splits.val,
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
            )

            write_path = os.path.join(self.opts.save_path, "geolife_val_data.beton")
            # Pass a type for each data field
            writer = DatasetWriter(
                write_path,
                {
                    # Tune options to optimize dataset size, throughput at train-time
                    "rgb": RGBImageField(max_resolution=256),
                    "near_ir": NDArrayField(
                        dtype=np.dtype("float32"), shape=(1, 256, 256)
                    ),
                    "label": IntField(),
                },
            )

            # Data decoding and augmentation (the first one is the left-most)
            img_pipeline = [
                CenterCropRGBImageDecoder(output_size=(224, 224), ratio=0.5),
                ImageMixup(alpha=0.5, same_lambda=True),
                ToTensor(),
                #                 ToDevice(0, non_blocking=True),
                ToTorchImage(),
                NormalizeImage(
                    np.array([106.9413, 114.8729, 104.5280]),
                    np.array([51.0005, 44.8594, 43.2014]),
                    np.float16,
                ),
            ]

            input_pipeline = [
                NDArrayDecoder(),
                ToTensor(),
                #                 ToDevice(0, non_blocking=True),
                transforms.Normalize([131.0458], [53.0884]),
            ]

            label_pipeline = [
                IntDecoder(),
                ToTensor(),
            ]
            #                 ToDevice(0, non_blocking=True)]

            # Pipeline for each data field
            pipelines = {
                "rgb": img_pipeline,
                "near_ir": input_pipeline,
                "label": label_pipeline,
            }

            val_loader = Loader(
                write_path,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                order=OrderOption.RANDOM,
                pipelines=pipelines,
            )
        else:
            val_dataset = GeoLifeCLEF2022Dataset(
                self.opts.dataset_path,
                self.opts.data.splits.val,
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=trf.get_transforms(
                    self.opts, "val"
                ),  # transforms.ToTensor(),
                target_transform=None,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                #             pin_memory=True,
            )
        return val_loader

    def test_dataloader(self):
        test_dataset = GeoLifeCLEF2022Dataset(
            self.opts.dataset_path,
            self.opts.data.splits.test,
            region="both",
            patch_data=self.opts.data.bands,
            use_rasters=False,
            patch_extractor=None,
            transform=trf.get_transforms(self.opts, "val"),  # transforms.ToTensor(),
            target_transform=None,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        return test_loader

    def train_one_epoch(self, epoch_idx):
        running_loss = 0.0
        last_loss = 0.0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for step, batch in enumerate(tqdm(self.train_loader)):
            # fetch the batch
            if self.opts.use_ffcv_loader:
                rgb_arr, nearIR_arr, target = batch
                input_patches = rgb_arr
                if "near_ir" in self.bands:
                    input_patches = torch.concatenate((rgb_arr, nearIR_arr), axis=0)

            else:
                patches, target, meta = batch
                input_patches = patches["input"]

            #             input_patches, target = input_patches.to(self.device), target.to(self.device)
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch and compute the loss
            outputs = None
            if self.opts.module.model == "inceptionv3":
                outputs, aux_outputs = self.forward(input_patches)
                loss1 = self.loss(outputs, target)
                loss2 = self.loss(aux_outputs, target)
                loss = loss1 + loss2
            else:
                outputs = self.forward(input_patches)
                loss = self.loss(outputs, target)

            # COMET LOGGING
            # self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
            # logging the metrics for training
            #             for (metric_name, _, scale) in self.metrics:
            #                 nname = "train_" + metric_name
            #                 metric_val = getattr(self, metric_name)(target, outputs)

            #                 self.log(nname, metric_val, on_step=True, on_epoch=True, sync_dist=True)

            # Compute the loss's gradients
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.detach().item()
            if step % 100 == 99:
                last_loss = running_loss / 100  # loss per batch
                print("  batch {} loss: {}".format(step + 1, last_loss))
                # COMET LOGGING
                #                 tb_x = epoch_index * len(training_loader) + step + 1
                #                 tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.0
        return last_loss

    def run_training_loop(self):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        # REPLACE WITH COMET
        #         writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0

        EPOCHS = 2

        best_vloss = 1_000_000.0

        for epoch in range(EPOCHS):
            print(f"EPOCH {epoch_number + 1}:")

            # Make sure gradient tracking is on, and do a pass over the data
            self.train(True)
            avg_loss = self.train_one_epoch(epoch_number)

            # We don't need gradients on to do reporting
            self.train(False)

            with torch.no_grad():
                running_vloss = 0.0
                for val_step, val_batch in enumerate(self.val_loader):
                    if self.opts.use_ffcv_loader:
                        rgb_arr, nearIR_arr, target = val_batch
                        input_patches = rgb_arr
                        if "near_ir" in self.bands:
                            input_patches = torch.concatenate(
                                (rgb_arr, nearIR_arr), axis=0
                            )

                    else:
                        patches, target, meta = val_batch
                        input_patches = patches["input"]

                    #                     input_patches, target = input_patches.to(self.device), target.to(self.device)

                    val_outputs = self.forward(input_patches)
                    val_loss = self.loss(val_outputs, target)
                    running_vloss += val_loss

                avg_vloss = running_vloss / (val_step + 1)
                print(f"LOSS train {avg_loss} valid {avg_vloss}")

                # COMET
                # Log the running loss averaged per batch
                # for both training and validation
                #             writer.add_scalars('Training vs. Validation Loss',
                #                             { 'Training' : avg_loss, 'Validation' : avg_vloss },
                #                             epoch_number + 1)
                # COMET
                #             writer.flush()

                # Track best performance, and save the model's state
                if avg_vloss < best_vloss:
                    best_vloss = avg_vloss
            #                 model_path = 'model_{}_{}'.format(epoch_number)
            #                 torch.save(model.state_dict(), model_path)

            epoch_number += 1

    def validation_step(self, batch, batch_idx):
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
