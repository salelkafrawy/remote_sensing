import os
import sys
import inspect
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, CURR_DIR)

from common import load_patch


class GeoLifeCLEF2022Dataset(Dataset):
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
        use_ffcv_loader,
        *,
        region="both",
        patch_data="all",
        use_rasters=True,
        patch_extractor=None,
        transform=None,
        target_transform=None,
        opts=None
    ):
        self.root = Path(root)
        self.subset = subset
        self.region = region
        self.patch_data = patch_data
        self.transform = transform
        self.target_transform = target_transform
        self.opts = opts

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

        # add country token
        df_fr["country"] = 0
        df_us["country"] = 1

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
        self.country = df["country"].values

        if self.training_data:
            self.targets = df["species_id"].values
        else:
            self.targets = None

        # FIXME: add back landcover one hot encoding?
        # self.one_hot_size = 34
        # self.one_hot = np.eye(self.one_hot_size)

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

        self.is_env_vars = False
        if self.opts:
            if self.opts.task == "multimodal":
                env_df = pd.read_csv(
                    self.root / "pre-extracted" / "environmental_vectors.csv",
                    sep=";",
                    index_col="observation_id",
                )
                env_df.fillna(np.finfo(np.float32).min, inplace=True)
                self.is_env_vars = True

                # load the standard scaler
                scaler_file = self.opts.scaler_file
                scaler_path = os.path.join(self.opts.data_dir, scaler_file)
                std_scaler = joblib.load(scaler_path)

                idx_col = env_df.index
                env_vars_normalized = std_scaler.transform(env_df.values)
                self.env_vars_df = pd.DataFrame(env_vars_normalized, index=idx_col)

        self.use_ffcv_loader = use_ffcv_loader

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, index):
        latitude = self.coordinates[index][0]
        longitude = self.coordinates[index][1]
        observation_id = self.observation_ids[index]
        country = self.country[index]
        meta = {
            "obs_id": observation_id,
            "lat": latitude,
            "lon": longitude,
            "country": country,
        }
        patches = load_patch(
            observation_id, self.root, self.use_ffcv_loader, data=self.patch_data
        )

        if self.training_data:
            target = self.targets[index]
            if self.target_transform:
                target = self.target_transform(target)

        if self.use_ffcv_loader:
            return patches["rgb"], target
        else:
            for s in patches:
                patches[s] = patches[s].squeeze(0)
                if self.is_env_vars:
                    patches["env_vars"] = self.env_vars_df.loc[
                        self.observation_ids[index]
                    ].values
                return patches, target, meta
            else:
                if self.is_env_vars:
                    patches["env_vars"] = self.env_vars_df.loc[
                        self.observation_ids[index]
                    ].values
                return patches, meta
