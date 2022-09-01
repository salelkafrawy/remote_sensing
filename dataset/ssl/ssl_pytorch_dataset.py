import os
import sys
import inspect

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, CURR_DIR)


import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pl_bolts.models.self_supervised.moco.transforms import GaussianBlur

from ssl_common import load_patch



class GeoLifeCLEF2022DatasetSSL(Dataset):
    """Pytorch dataset handler for GeoLifeCLEF 2022 dataset.

    Parameters
    ----------
    root : string or pathlib.Path
        Root directory of dataset.
    subset : string, either "train", "val", "train+val" or "test"
        Use the given subset ("train+val" is the complete training data).
    region : string, either "both", "fr" or "us"
        Load the observations of both France and US or only a single region.
    patch_data : string or list of string
        Specifies what type of patch data to load, possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'.
    use_rasters : boolean (optional)
        If True, extracts patches from environmental rasters.
    patch_extractor : PatchExtractor object (optional)
        Patch extractor to use if rasters are used.
    transform : callable (optional)
        A function/transform that takes a list of arrays and returns a transformed version.
    target_transform : callable (optional)
        A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        root,
        use_ffcv_loader,
        *,
        region="both",
        patch_data="all",
        use_rasters=True,
        patch_extractor=None,
        transform=None,
        target_transform=None
    ):
        self.root = Path(root)
        self.region = region
        self.patch_data = patch_data
        self.transform = transform
        self.target_transform = target_transform

        possible_regions = ["both", "fr", "us"]
        if region not in possible_regions:
            raise ValueError(
                "Possible values for 'region' are: {} (given {})".format(
                    possible_regions, region
                )
            )

        # load the training data
        df_train_fr = pd.read_csv(
            self.root / "observations" / "observations_fr_train.csv",
            sep=";",
            index_col="observation_id",
        )

        df_train_us = pd.read_csv(
            self.root / "observations" / "observations_us_train.csv",
            sep=";",
            index_col="observation_id",
        )
        # keep only the train (apart from the val set)
        df_train_val = pd.concat((df_train_fr, df_train_us))
        ind_train = df_train_val.index[df_train_val["subset"] == "train"]
        df_train = df_train_val.loc[ind_train]

        # load the test data
        df_test_fr = pd.read_csv(
            self.root / "observations" / "observations_fr_test.csv",
            sep=";",
            index_col="observation_id",
        )
        df_test_us = pd.read_csv(
            self.root / "observations" / "observations_us_test.csv",
            sep=";",
            index_col="observation_id",
        )
        df_test = pd.concat((df_test_fr, df_test_us))

        # concatenate train and test data
        df = pd.concat((df_train, df_test))

        # for debugging:
        #         df = df_test_fr.iloc[:1024]
        self.observation_ids = df.index

        self.use_ffcv_loader = use_ffcv_loader

    def __len__(self):
        return len(self.observation_ids)

    augment_1 = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])
    
    augment_2 = transforms.Compose([
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
    ])
    
    augment_3 = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
    ])
    
    preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.ToTensor(),
        transforms.Normalize([0.4194, 0.4505, 0.4099], [0.20, 0.1759, 0.1694]),
    ])
    
    def __getitem__(self, index):

        observation_id = self.observation_ids[index]

        patches = load_patch(
            observation_id, self.root, self.use_ffcv_loader, data=self.patch_data
        )

        q = patches
        k0 = self.augment_1(q)
        k1 = self.augment_2(q)
        k2 = self.augment_3(q)

        q = self.preprocess(q)
        k0 = self.preprocess(k0)
        k1 = self.preprocess(k1)
        k2 = self.preprocess(k2)
        
        return q, [k0, k1, k2]
