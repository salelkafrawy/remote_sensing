import numpy as np
from torchvision import transforms

import ffcv
import torch
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


# weight of size [64, 3, 7, 7], expected input[64, 256, 256, 3] to have 3 channels, but got 256 channels instead
# Data decoding and augmentation
rgb_pipeline =  [
    
    CenterCropRGBImageDecoder(output_size=(224, 224), ratio=0.5),
    RandomHorizontalFlip(flip_prob=0.5),
    ImageMixup(alpha=0.5, same_lambda=True),
    ToTensor(),
#     ToDevice( torch.device(this_device), non_blocking=True),
#     ToTorchImage(),
    ffcv.transforms.ToTorchImage(channels_last=False, convert_back_int16=False),
    ffcv.transforms.Convert(torch.float32),
    NormalizeImage(
        np.array([106.9413, 114.8729, 104.5280]),
        np.array([51.0005, 44.8594, 43.2014]),
        np.float32,
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
#     CenterCropRGBImageDecoder(output_size=(224, 224), ratio=0.5),
#     RandomHorizontalFlip(flip_prob=0.5),
#     ImageMixup(alpha=0.5, same_lambda=True),
#     ToTensor(),
# #     ToDevice(0, non_blocking=True),
#     ToTorchImage(),
#     NormalizeImage(
#         np.array([106.9413, 114.8729, 104.5280]),
#         np.array([51.0005, 44.8594, 43.2014]),
#         np.float32,
#     ),
# ]

near_ir_pipeline = [
    NDArrayDecoder(),
    ToTensor(),
#     ToDevice(0, non_blocking=True),
    transforms.Normalize(np.array([131.0458]), np.array([53.0884])),
]

landcover_pipeline = [
    NDArrayDecoder(),
    ToTensor(),
#     ToDevice(0, non_blocking=True),
    transforms.Normalize(np.array([17.4200]), np.array([9.5173])),
]

altitude_pipeline = [
    NDArrayDecoder(),
    ToTensor(),
#     ToDevice(0, non_blocking=True),
    transforms.Normalize(np.array([298.1693]), np.array([459.3285])),
]

label_pipeline = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze()] #, ToDevice(0, non_blocking=True)]

# Pipeline for each data field
FFCV_PIPELINES = {"image": rgb_pipeline, 
#              "near_ir": near_ir_pipeline,
#              "altitude": altitude_pipeline,
#              "landcover": landcover_pipeline,
             "label": label_pipeline}