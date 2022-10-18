import os
from typing import Any, Dict, Optional
import numpy as np
import logging

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from .pytorch_dataset import GeoLifeCLEF2022Dataset
# import ffcv
# from ffcv.loader import Loader, OrderOption
# from ffcv.fields import RGBImageField, IntField, NDArrayField
# from ffcv.writer import DatasetWriter
# from composer.datasets.ffcv_utils import write_ffcv_dataset
# from composer.datasets.ffcv_utils import ffcv_monkey_patches

from .utils import FFCV_PIPELINES

log = logging.getLogger(__name__)


class GeoLifeDataModule(pl.LightningDataModule):
    def __init__(self, opts, **kwargs: Any):
        super().__init__()
        self.opts = opts
        self.data_dir = self.opts.data_dir

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if self.opts.use_ffcv_loader:
            train_dataset = GeoLifeCLEF2022Dataset(
                self.opts.data_dir,
                self.opts.data.splits.train,
                self.opts.use_ffcv_loader,
                region="both",
                patch_data="all",  # self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
            )

            self.train_write_path = os.path.join(
                self.opts.ffcv_write_path, "geolife_train_data.ffcv"
            )
            write_ffcv_dataset(dataset=train_dataset, write_path=self.train_write_path)

            val_dataset = GeoLifeCLEF2022Dataset(
                self.opts.data_dir,
                self.opts.data.splits.val,
                self.opts.use_ffcv_loader,
                region="both",
                patch_data="all",  # self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
            )

            self.val_write_path = os.path.join(
                self.opts.ffcv_write_path, "geolife_val_data.ffcv"
            )
            write_ffcv_dataset(dataset=val_dataset, write_path=self.val_write_path)

            ffcv_monkey_patches()
        else:

            # data and transforms
            self.train_dataset = GeoLifeCLEF2022Dataset(
                self.data_dir,
                self.opts.data.splits.train,
                self.opts.use_ffcv_loader,
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
                opts=self.opts,
            )

            self.val_dataset = GeoLifeCLEF2022Dataset(
                self.data_dir,
                self.opts.data.splits.val,
                self.opts.use_ffcv_loader,
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
                opts=self.opts,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = GeoLifeCLEF2022Dataset(
                self.opts.data_dir,
                self.opts.data.splits.test,
                False,  # self.opts.use_ffcv_loader,
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
                opts=self.opts,
            )

        if stage == "predict" or stage is None:
            self.test_dataset = GeoLifeCLEF2022Dataset(
                self.opts.data_dir,
                self.opts.data.splits.test,
                False,  # self.opts.use_ffcv_loader,
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
                opts=self.opts,
            )

    def train_dataloader(self):
        if self.opts.use_ffcv_loader:
            train_loader = Loader(
                self.train_write_path,
                batch_size=self.opts.data.loaders.batch_size,
                num_workers=self.opts.data.loaders.num_workers,
                order=OrderOption.RANDOM,
                pipelines=FFCV_PIPELINES,
            )
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.opts.data.loaders.batch_size,
                num_workers=self.opts.data.loaders.num_workers,
                shuffle=True,
                pin_memory=True,
            )
        return train_loader

    def val_dataloader(self):

        if self.opts.use_ffcv_loader:
            val_loader = Loader(
                self.val_write_path,
                batch_size=self.opts.data.loaders.batch_size,
                num_workers=self.opts.data.loaders.num_workers,
                order=OrderOption.RANDOM,
                pipelines=FFCV_PIPELINES,
            )
        else:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.opts.data.loaders.batch_size,
                num_workers=self.opts.data.loaders.num_workers,
                shuffle=False,
                pin_memory=True,
            )
        return val_loader

    def test_dataloader(self):

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.opts.data.loaders.batch_size,
            num_workers=self.opts.data.loaders.num_workers,
            shuffle=False,
        )

        return test_loader

    def predict_dataloader(self):

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.opts.data.loaders.batch_size,
            num_workers=self.opts.data.loaders.num_workers,
            shuffle=False,
        )

        return test_loader
