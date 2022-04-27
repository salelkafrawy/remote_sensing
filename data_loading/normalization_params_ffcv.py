import os
import sys
import inspect
import numpy as np
import pandas as pd


CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, CURR_DIR)
sys.path.insert(0, PARENT_DIR)

from pathlib import Path
import timeit
from tqdm import tqdm
import torch 
from dataset_ffcv import GeoLifeCLEF2022Dataset
from torch.utils.data import DataLoader

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, NDArrayField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage


BAND = "near_ir"
REGION = "both"  # us, fr, both


if __name__ == "__main__":
    
    print("loading dataset ...")
    dataset = GeoLifeCLEF2022Dataset("/network/scratch/s/sara.ebrahim-elkafrawy/",
        subset= "train",
        region= REGION,
        patch_data="near_ir",
        use_rasters=False,
        patch_extractor=None,
        transform=None, # transform=transforms.Compose([transforms.ToTensor()])
        target_transform=None)
    

    write_path = "/home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/geolife_nearIR_train_data.beton"
    # Pass a type for each data field
    writer = DatasetWriter(write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'image': NDArrayField(shape=(1,256,256)),
    'label': IntField()
    })

    # Write dataset
    writer.from_indexed_dataset(dataset)

    # Data decoding and augmentation
    image_pipeline = [ToTensor(), ToTorchImage(), ToDevice(0)]
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(0)]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }

    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    loader = Loader(write_path, batch_size=32, num_workers=0,
                    order=OrderOption.RANDOM, pipelines=pipelines)
    

    print("finished writing the dataset")
    

    start = timeit.default_timer()
    
    nearIR_mean, nearIR_std = calculate_nearIR_params()
#     rgb_mean, rgb_std = calculate_rgb_params()

    stop = timeit.default_timer()

    print("Elapsed calculation time: ", stop - start)
    
    print(f"mean: {mean} ,std: {std}")


def calculate_rgb_params():
    
        print("loading dataset ...")
    dataset = GeoLifeCLEF2022Dataset("/network/scratch/s/sara.ebrahim-elkafrawy/",
        subset= "train",
        region= REGION,
        patch_data="rgb",
        use_rasters=False,
        patch_extractor=None,
        transform=None, # transform=transforms.Compose([transforms.ToTensor()])
        target_transform=None)
    

    write_path = "/home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/geolife_rgb_train_data.beton"
    # Pass a type for each data field
    writer = DatasetWriter(write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'image': RGBImageField(
        max_resolution=256,
    ),
    'label': IntField()
    })

    # Write dataset
    writer.from_indexed_dataset(dataset)

    # Data decoding and augmentation
    image_pipeline = [ToTensor(), ToTorchImage(), ToDevice(0)]
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(0)]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }

    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    loader = Loader(write_path, batch_size=32, num_workers=0,
                    order=OrderOption.RANDOM, pipelines=pipelines)
    

    
    num_of_pixels = len(dataset) * 256 * 256
    # calculate the mean
    total_sum = 0
    for batch in tqdm(loader):
        total_sum += batch[0].sum((0,1,2))
    mean = total_sum / num_of_pixels

    # US region mean = torch.Tensor([112.3288, 121.6368, 113.5514])
    print(f"mean: {mean}")
    
    # calculate the std
    sum_of_squared_error = 0
    for batch in tqdm(loader): 
        sum_of_squared_error += ((batch[0] - mean[None, None, None, :]).pow(2)).sum((0,1,2))
    std = torch.sqrt(sum_of_squared_error / num_of_pixels)