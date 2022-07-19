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
from common_ffcv import load_patch


class GeoLifeCLEF2022DatasetFFCV(Dataset):
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
        subset,
        *,
        region="both",
        patch_data="all",
        use_rasters=True,
        patch_extractor=None,
        transform=None,
        target_transform=None
    ):
        self.root = Path(root)
        self.subset = subset
        self.region = region
        self.patch_data = patch_data
        self.transform = transform
        self.target_transform = target_transform

        possible_subsets = ["train", "val", "train+val", "test"]
        if subset not in possible_subsets:
            raise ValueError(
                "Possible values for 'subset' are: {} (given {})".format(
                    possible_subsets, subset
                )
            )

        possible_regions = ["both", "fr", "us"]
        if region not in possible_regions:
            raise ValueError(
                "Possible values for 'region' are: {} (given {})".format(
                    possible_regions, region
                )
            )

        if subset == "test":
            subset_file_suffix = "test"
            self.training_data = False

        else:
            subset_file_suffix = "train"
            self.training_data = True

        df_fr = pd.read_csv(
            self.root
            / "observations"
            / "observations_fr_{}.csv".format(subset_file_suffix),
            sep=";",
            index_col="observation_id",
        )
        df_us = pd.read_csv(
            self.root
            / "observations"
            / "observations_us_{}.csv".format(subset_file_suffix),
            sep=";",
            index_col="observation_id",
        )

        if region == "both":
            df = pd.concat((df_fr, df_us))
        elif region == "fr":
            df = df_fr
        elif region == "us":
            df = df_us

        if self.training_data and subset != "train+val":
            ind = df.index[df["subset"] == subset]
            df = df.loc[ind]

        # for debugging
#         df = df.iloc[:1024]

        self.observation_ids = df.index
        self.coordinates = df[["latitude", "longitude"]].values

        if self.training_data:
            self.targets = df["species_id"].values
        else:
            self.targets = None

        if use_rasters:
            if patch_extractor is None:
                from .environmental_raster import PatchExtractor

                patch_extractor = PatchExtractor(
                    self.root / "rasters", size=256
                )  # check resolution of the patches
                patch_extractor.add_all_rasters()

            self.patch_extractor = patch_extractor
        else:
            self.patch_extractor = None

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, index):

        latitude = self.coordinates[index][0]
        longitude = self.coordinates[index][1]
        observation_id = self.observation_ids[index]
        meta = (observation_id, latitude, longitude)

        rgb_arr, nearIR_arr, altitude_arr, landcover_arr = load_patch(
            observation_id, self.root, data=self.patch_data
        )

        if self.training_data:
            target = self.targets[index]

            if self.target_transform:
                target = self.target_transform(target)

            #             tmp_arr = np.zeros()
            #             for idx in range(1, len(self.bands)):
            #                 patches["input"] = torch.cat(
            #                     (patches["input"], patches[self.bands[idx]]), axis=1
            #                 )

            #             print(f"target: {type(target)} .. rgb_arr:{type(rgb_arr)}")
            return rgb_arr, target
        # nearIR_arr, altitude_arr, landcover_arr
        else:

            return rgb_arr
