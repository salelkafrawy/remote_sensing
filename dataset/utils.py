import os
import sys
import inspect
import numpy as np
import torch
from torchvision import transforms


CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, CURR_DIR)




# import ffcv
# from ffcv.fields.decoders import (
#     IntDecoder,
#     NDArrayDecoder,
#     SimpleRGBImageDecoder,
#     CenterCropRGBImageDecoder,
# )
# from ffcv.loader import Loader, OrderOption

# from ffcv.transforms import  NormalizeImage,  # RAISES A NUMBA ERROR


# from composer.datasets.ffcv_utils import ffcv_monkey_patches
# from composer.datasets.ffcv_utils import write_ffcv_dataset

from pytorch_dataset import GeoLifeCLEF2022Dataset


# Data decoding and augmentation
# rgb_pipeline = [
#     CenterCropRGBImageDecoder(output_size=(224, 224), ratio=0.5),
#     ffcv.transforms.RandomHorizontalFlip(flip_prob=0.5),
#     ffcv.transforms.ImageMixup(alpha=0.5, same_lambda=True),
#     ffcv.transforms.ToTensor(),
#     ffcv.transforms.ToTorchImage(channels_last=False, convert_back_int16=False),
#     ffcv.transforms.Convert(torch.float32),
#     transforms.Normalize(
#         np.array([106.9413, 114.8729, 104.5280]),
#         np.array([51.0005, 44.8594, 43.2014]),
#     ),
# ]

# near_ir_pipeline = [
#     NDArrayDecoder(),
#     ffcv.transforms.ToTensor(),
#     transforms.Normalize(np.array([131.0458]), np.array([53.0884])),
# ]

# landcover_pipeline = [
#     NDArrayDecoder(),
#     ffcv.transforms.ToTensor(),
#     transforms.Normalize(np.array([17.4200]), np.array([9.5173])),
# ]

# altitude_pipeline = [
#     NDArrayDecoder(),
#     ffcv.transforms.ToTensor(),
#     transforms.Normalize(np.array([298.1693]), np.array([459.3285])),
# ]

# label_pipeline = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze()]

# Pipeline for each data field
# FFCV_PIPELINES = {
#     "image": rgb_pipeline,
#     #                  "near_ir": near_ir_pipeline,
#     #              "altitude": altitude_pipeline,
#     #              "landcover": landcover_pipeline,
#     "label": label_pipeline,
# }
