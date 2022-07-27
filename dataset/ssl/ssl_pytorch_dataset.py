import os
import sys
import inspect

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, CURR_DIR)


from pathlib import Path

import pandas as pd

from torch.utils.data import Dataset
import numpy as np
from dataset.common import load_patch


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

    def __getitem__(self, index):

        observation_id = self.observation_ids[index]

        patches = load_patch(
            observation_id, self.root, self.use_ffcv_loader, data=self.patch_data
        )

        if self.use_ffcv_loader:
            return patches["rgb"], 0
        else:
            for s in patches:
                patches[s] = patches[s].squeeze(0)
            return patches
