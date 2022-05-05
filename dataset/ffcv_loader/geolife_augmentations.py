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
from torchvision import transforms
from dataset_ffcv import GeoLifeCLEF2022DatasetFFCV
from torch.utils.data import DataLoader
from PIL import Image

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, NDArrayField
from ffcv.fields.decoders import IntDecoder, NDArrayDecoder, SimpleRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import RandomHorizontalFlip, Cutout, NormalizeImage,\
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, ImageMixup


BAND = ["rgb", "near_ir"]  # near_ir (uint8) landcover (uint8) altitude (int16)
REGION = "both"  # us, fr, both
SUBSET = "val"

if __name__ == "__main__":
    
    print("loading dataset ...")
    dataset = GeoLifeCLEF2022DatasetFFCV("/network/scratch/s/sara.ebrahim-elkafrawy/",
        subset= SUBSET,
        region= REGION,
        patch_data=BAND,
        use_rasters=False,
        patch_extractor=None,
        transform=None, # transform=transforms.Compose([transforms.ToTensor()])
        target_transform=None)
    

    write_path = f"/home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/geolife_nearIR_{SUBSET}_data.beton"
    # Pass a type for each data field
    writer = DatasetWriter(write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'rgb': RGBImageField(max_resolution=256),
    'near_ir': NDArrayField(dtype=np.dtype('float32'), shape=(1,256,256)),
    'label': IntField()
    })

    # Write dataset
    writer.from_indexed_dataset(dataset)

    # Data decoding and augmentation (the first one is the left-most)
    img_pipeline = [
        CenterCropRGBImageDecoder(output_size=(224,224), ratio=0.5),
        RandomHorizontalFlip(flip_prob=0.5),
        ImageMixup(alpha=0.5, same_lambda=True),
        ToTensor(), 
        ToDevice(0, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(np.array([106.9413, 114.8729, 104.5280]), np.array([51.0005, 44.8594, 43.2014]), np.float16),
    ]
    
    input_pipeline = [
        NDArrayDecoder(),
        ToTensor(), 
        ToDevice(0, non_blocking=True),
        transforms.Normalize([131.0458], [53.0884]),
    ]
    
    label_pipeline = [
        IntDecoder(), 
        ToTensor(),
        ToDevice(0, non_blocking=True)]

    # Pipeline for each data field
    pipelines = {
        'rgb': img_pipeline,
        'near_ir': input_pipeline,
        'label': label_pipeline
    }

    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    loader = Loader(write_path, batch_size=32, num_workers=2,
                    order=OrderOption.RANDOM, pipelines=pipelines)
    
    print("finished writing the dataset")
    
    batch = next(iter(loader))
    from IPython import embed
    embed(header="check a batch from loader and see the shapes")
    


