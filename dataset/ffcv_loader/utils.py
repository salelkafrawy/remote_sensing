import os
import numpy as np

import torch
from torchvision import transforms

import ffcv
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
    #     NormalizeImage,  # RAISES A NUMBA ERROR
    RandomTranslate,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
    ImageMixup,
)

from composer.datasets.ffcv_utils import ffcv_monkey_patches
from composer.datasets.ffcv_utils import write_ffcv_dataset

from dataset_ffcv import GeoLifeCLEF2022DatasetFFCV


# Data decoding and augmentation
rgb_pipeline = [
    CenterCropRGBImageDecoder(output_size=(224, 224), ratio=0.5),
    RandomHorizontalFlip(flip_prob=0.5),
    ImageMixup(alpha=0.5, same_lambda=True),
    ToTensor(),
    ffcv.transforms.ToTorchImage(channels_last=False, convert_back_int16=False),
    ffcv.transforms.Convert(torch.float32),
    transforms.Normalize(
        np.array([106.9413, 114.8729, 104.5280]),
        np.array([51.0005, 44.8594, 43.2014]),
    ),
]
#                 SimpleRGBImageDecoder(),
#                 ffcv.transforms.ToTensor(),
#                 ffcv.transforms.ToTorchImage(),
#                 ffcv.transforms.Convert(torch.float32),
#                 transforms.Normalize(
#                     np.array([106.9413, 114.8729, 104.5280]),
#                     np.array([51.0005, 44.8594, 43.2014]),),
# ]


near_ir_pipeline = [
    NDArrayDecoder(),
    ToTensor(),
    transforms.Normalize(np.array([131.0458]), np.array([53.0884])),
]

landcover_pipeline = [
    NDArrayDecoder(),
    ToTensor(),
    transforms.Normalize(np.array([17.4200]), np.array([9.5173])),
]

altitude_pipeline = [
    NDArrayDecoder(),
    ToTensor(),
    transforms.Normalize(np.array([298.1693]), np.array([459.3285])),
]

label_pipeline = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze()]

# Pipeline for each data field
FFCV_PIPELINES = {
    "image": rgb_pipeline,
#                  "near_ir": near_ir_pipeline,
    #              "altitude": altitude_pipeline,
    #              "landcover": landcover_pipeline,
    "label": label_pipeline,
}


def get_ffcv_dataloaders(exp_configs):

    train_dataset = GeoLifeCLEF2022DatasetFFCV(
        exp_configs.data_dir,
        exp_configs.data.splits.train,
        region="both",
        patch_data="all",  # self.opts.data.bands,
        use_rasters=False,
        patch_extractor=None,
        transform=None,
        target_transform=None,
    )

    train_write_path = os.path.join(exp_configs.ffcv_write_path, "geolife_train_data.ffcv")
    write_ffcv_dataset(dataset=train_dataset, write_path=train_write_path)

    val_dataset = GeoLifeCLEF2022DatasetFFCV(
        exp_configs.data_dir,
        exp_configs.data.splits.val,
        region="both",
        patch_data="all",  # self.opts.data.bands,
        use_rasters=False,
        patch_extractor=None,
        transform=None,
        target_transform=None,
    )

    val_write_path = os.path.join(exp_configs.ffcv_write_path, "geolife_val_data.ffcv")
    write_ffcv_dataset(dataset=val_dataset, write_path=val_write_path)

    ffcv_monkey_patches()

    train_loader = Loader(
        train_write_path,
        batch_size=exp_configs.data.loaders.batch_size,
        num_workers=exp_configs.data.loaders.num_workers,
        order=OrderOption.RANDOM,
        pipelines=FFCV_PIPELINES,
    )
    val_loader = Loader(
        val_write_path,
        batch_size=exp_configs.data.loaders.batch_size,
        num_workers=exp_configs.data.loaders.num_workers,
        order=OrderOption.RANDOM,
        pipelines=FFCV_PIPELINES,
    )
    return train_loader, val_loader
