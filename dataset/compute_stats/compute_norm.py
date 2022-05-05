import os
import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path().resolve().parent))
sys.path.append(str(Path().resolve().parent.parent))

import transforms.transforms as trf
import pandas as pd

from pytorch_dataset import GeoLifeCLEF2022Dataset

def main_means():
    print("loading dataset")
    dataset = GeoLifeCLEF2022Dataset("/network/scratch/s/sara.ebrahim-elkafrawy/",
        subset= "train",
        region="us",
        patch_data="near_ir",
        use_rasters=False,
        patch_extractor=None,
        transform=None,
        target_transform=None)
    print("finished loading dataset")

    means = np.zeros(1)
    std = np.zeros(3)
    count = 0
    for elem in dataset: 
        patch = elem[0]
        temp = np.mean(patch)
        means = (means*count + temp)/(count + 1)
        count += 1
        if count % 2500 == 0:
            print(count)
        
    print("done")
    print(count)
    np.save("/network/scratch/s/sara.ebrahim-elkafrawy/normalization/nir_us.npy", means)

def main_std():
    print("loading dataset")
    dataset = GeoLifeCLEF2022Dataset("/network/scratch/s/sara.ebrahim-elkafrawy/",
        subset= "train",
        region="us",
        patch_data="rgb",
        use_rasters=False,
        patch_extractor=None,
        transform=None,
        target_transform=None)
    print("finished loading dataset")

    means = np.load("/network/scratch/s/sara.ebrahim-elkafrawy/normalization/rgb_us.npy")
    means = np.expand_dims(means, 1)
    means = np.expand_dims(means, 1)
    means = np.repeat(means, 256, axis = 1)
    means = np.repeat(means, 256, axis = 2)
    std = np.zeros(3)
    count = 0
    for elem in dataset: 
        #patch = (elem[0]["nir"] - means)**2
        patch = (elem[0]["rgb"] - means)**2
        temp = np.mean(patch.numpy(), axis = (-1,-2))[0]
        std = (std*count + temp)/(count + 1)
        count += 1
        if count % 2500 == 0:
            print(count)
        
    print("done")
    print(count)
    np.save("/network/scratch/s/sara.ebrahim-elkafrawy/normalization/rgb_us_std.npy", std)
 

if __name__ == "__main__":
    #main_means()
    main_std()
