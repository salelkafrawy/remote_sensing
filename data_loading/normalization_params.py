import os
import sys
import inspect
import numpy as np
import pandas as pd
from pathlib import Path
import timeit
from tqdm import tqdm
import torch 

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, CURR_DIR)
sys.path.insert(0, PARENT_DIR)

from pytorch_dataset import GeoLifeCLEF2022Dataset
from torch.utils.data import DataLoader
# import transforms.transforms as trf


BAND = "rgb"
REGION = "both"  # us, fr, both

if __name__ == "__main__":
    
    print("loading dataset ...")
    dataset = GeoLifeCLEF2022Dataset("/network/scratch/s/sara.ebrahim-elkafrawy/",
        subset= "train",
        region= REGION,
        patch_data=BAND,
        use_rasters=False,
        patch_extractor=None,
        transform=None, # transform=transforms.Compose([transforms.ToTensor()])
        target_transform=None)
    print("finished loading dataset")
    
    loader = DataLoader(dataset, batch_size=32, num_workers=2)
    # data = next(iter(loader))
    
    
    num_of_pixels = len(dataset) * 256 * 256
    
    start = timeit.default_timer()
    # calculate the mean
    total_sum = 0
    for batch in tqdm(loader):
        total_sum += batch[0][BAND].sum((0,2,3))
    mean = total_sum / num_of_pixels

    # US region mean = torch.Tensor([112.3288, 121.6368, 113.5514])
    print(f"mean: {mean}")
    
    # calculate the std
    sum_of_squared_error = 0
    for batch in tqdm(loader): 
        sum_of_squared_error += ((batch[0][BAND] - mean[None, :, None, None]).pow(2)).sum((0,2,3))
    std = torch.sqrt(sum_of_squared_error / num_of_pixels)

    stop = timeit.default_timer()

    print("Elapsed calculation time: ", stop - start)
    
    print(f"mean: {mean} ,std: {std}")

