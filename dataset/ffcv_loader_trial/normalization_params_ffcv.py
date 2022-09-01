import os
import sys
import inspect
import numpy as np
import pandas as pd
import logging
log = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, CURR_DIR)
sys.path.insert(0, PARENT_DIR)

from pathlib import Path
import timeit
from tqdm import tqdm
import torch
from dataset_ffcv import GeoLifeCLEF2022DatasetFFCV
from torch.utils.data import DataLoader

from torchvision import transforms
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, NDArrayField
from ffcv.fields.decoders import (
    IntDecoder,
    NDArrayDecoder,
    SimpleRGBImageDecoder,
    CenterCropRGBImageDecoder,
)
import ffcv
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


BAND = ["rgb", "near_ir", "altitude", "landcover"]  # near_ir (uint8) landcover (uint8) altitude (int16)
REGION = "both"  # us, fr, both
SUBSET = "val"  


def calculate_rgb_params():

    print("loading dataset ...")
    dataset = GeoLifeCLEF2022DatasetFFCV(
        "/network/scratch/s/sara.ebrahim-elkafrawy/",
        subset=SUBSET,
        region=REGION,
        patch_data=BAND,
        use_rasters=False,
        patch_extractor=None,
        transform=None,  # transform=transforms.Compose([transforms.ToTensor()])
        target_transform=None,
    )

    write_path = f"/home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/geolife_rgb_{SUBSET}_data.beton"
    log.info(f'Writing dataset in FFCV <file>.ffcv format to {write_path}.')
    writer = ffcv.writer.DatasetWriter(write_path, {
        'rgb':
            ffcv.fields.RGBImageField(write_mode='raw',
                                      max_resolution=256,
                                      compress_probability=0.5,
                                      jpeg_quality=90),
        'label':
            ffcv.fields.IntField()
    },
        num_workers=0)

    writer.from_indexed_dataset(dataset, chunksize=100)
            
            
#     # Pass a type for each data field
#     writer = DatasetWriter(
#         write_path,
#         {
#             # Tune options to optimize dataset size, throughput at train-time
#             # Tune options to optimize dataset size, throughput at train-time
#             "rgb": RGBImageField(max_resolution=256),
# #             "near_ir": NDArrayField(dtype=np.dtype("float32"), shape=(1, 256, 256)),
# #             "altitude": NDArrayField(dtype=np.dtype("float32"), shape=(1, 256, 256)),
# #             "landcover": NDArrayField(dtype=np.dtype("float32"), shape=(1, 256, 256)),
#             "label": IntField(),
#         },
#     )

#     # Write dataset
#     writer.from_indexed_dataset(dataset)

    # Data decoding and augmentation
    rgb_pipeline =  [
        CenterCropRGBImageDecoder(output_size=(224, 224), ratio=0.5),
        RandomHorizontalFlip(flip_prob=0.5),
        ImageMixup(alpha=0.5, same_lambda=True),
        ToTensor(),
        ToDevice(0, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(
            np.array([106.9413, 114.8729, 104.5280]),
            np.array([51.0005, 44.8594, 43.2014]),
            np.float32,
        ),
    ]
    
    near_ir_pipeline = [
        NDArrayDecoder(),
        ToTensor(),
        ToDevice(0, non_blocking=True),
        transforms.Normalize([131.0458], [53.0884]),
    ]
    
    landcover_pipeline = [
        NDArrayDecoder(),
        ToTensor(),
        ToDevice(0, non_blocking=True),
        transforms.Normalize([17.4200], [9.5173]),
    ]
    
    altitude_pipeline = [
        NDArrayDecoder(),
        ToTensor(),
        ToDevice(0, non_blocking=True),
        transforms.Normalize([298.1693], [459.3285]),
    ]
        
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(0, non_blocking=True)]

    # Pipeline for each data field
    pipelines = {"rgb": rgb_pipeline, 
#                  "near_ir": near_ir_pipeline,
#                  "altitude": altitude_pipeline,
#                  "landcover": landcover_pipeline,
                 "label": label_pipeline}

    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    loader = Loader(
        write_path,
        batch_size=32,
        num_workers=2,
        order=OrderOption.RANDOM,
        pipelines=pipelines,
    )

    num_of_pixels = len(dataset) * 256 * 256
    # calculate the mean
    
    from IPython import embed
    embed(header='check loader')
    
    total_sum = 0
    for batch in tqdm(loader):
        total_sum += batch[0].sum((0, 2, 3))
    mean = total_sum / num_of_pixels

    # US region mean = torch.Tensor([112.3288, 121.6368, 113.5514])
    print(f"mean: {mean}")

    # calculate the std
    sum_of_squared_error = 0
    for batch in tqdm(loader):
        sum_of_squared_error += ((batch[0] - mean[None, :, None, None]).pow(2)).sum(
            (0, 2, 3)
        )
    std = torch.sqrt(sum_of_squared_error / num_of_pixels)
    
    return mean, std

    
if __name__ == "__main__":

    start = timeit.default_timer()

    rgb_mean, rgb_std = calculate_rgb_params()

    stop = timeit.default_timer()

    print("Elapsed calculation time: ", stop - start)

    print(f"mean: {rgb_mean} ,std: {rgb_std}")


