import os
from typing import Any, Dict, Optional
import numpy as np
import logging

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from .pytorch_dataset import GeoLifeCLEF2022Dataset
import ffcv
from ffcv.loader import Loader, OrderOption
from ffcv.fields import RGBImageField, IntField, NDArrayField
from ffcv.writer import DatasetWriter
from composer.datasets.ffcv_utils import write_ffcv_dataset
from composer.datasets.ffcv_utils import ffcv_monkey_patches

from .ffcv_loader.dataset_ffcv import GeoLifeCLEF2022DatasetFFCV
from .ffcv_loader.utils import FFCV_PIPELINES

log = logging.getLogger(__name__)



class GeoLifeDataModule(pl.LightningDataModule):
    def __init__(self, opts, **kwargs: Any):
        super().__init__()
        self.opts = opts
        self.data_dir = self.opts.data_dir

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if self.opts.use_ffcv_loader:

            train_dataset = GeoLifeCLEF2022DatasetFFCV(
                self.data_dir,
                self.opts.data.splits.train,
                region="both",
                patch_data="all", # self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
            )

            self.train_write_path = os.path.join(
                self.opts.log_dir, "geolife_train_data.ffcv"
            )
            write_ffcv_dataset(dataset=train_dataset, write_path=self.train_write_path)
#             log.info(f'Writing train dataset in FFCV <file>.ffcv format to {self.train_write_path}.')
#             writer = ffcv.writer.DatasetWriter(self.train_write_path, {
#                 'image':
#                     ffcv.fields.RGBImageField(write_mode='raw',
#                                               max_resolution=256,
#                                               compress_probability=0.5,
#                                               jpeg_quality=90),
#                 'label':
#                     ffcv.fields.IntField()
#             },
#                 num_workers=self.opts.data.loaders.num_workers)
            
#             writer.from_indexed_dataset(train_dataset, chunksize=100)
    

#             train_writer = DatasetWriter(
#                 self.train_write_path,
#                 {
#                     # Tune options to optimize dataset size, throughput at train-time
#                     "rgb": RGBImageField(max_resolution=256),
#                     "near_ir": NDArrayField(dtype=np.dtype("float32"), shape=(1, 256, 256)),
#                     "altitude": NDArrayField(dtype=np.dtype("float32"), shape=(1, 256, 256)),
#                     "landcover": NDArrayField(dtype=np.dtype("float32"), shape=(1, 256, 256)),
#                     "label": IntField(),
#                 },
#                 num_workers=self.opts.data.loaders.num_workers
#             )
#             # Write dataset
#             train_writer.from_indexed_dataset(train_dataset)

            val_dataset = GeoLifeCLEF2022DatasetFFCV(
                self.data_dir,
                self.opts.data.splits.val,
                region="both",
                patch_data="all", #self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
            )
    
            self.val_write_path = os.path.join(
                self.opts.log_dir, "geolife_val_data.ffcv"
            )
            write_ffcv_dataset(dataset=val_dataset, write_path=self.val_write_path)
#             log.info(f'Writing val dataset in FFCV <file>.ffcv format to {self.val_write_path}.')
#             val_writer = ffcv.writer.DatasetWriter(self.val_write_path, {
#                 'image':
#                     ffcv.fields.RGBImageField(write_mode='raw',
#                                               max_resolution=256,
#                                               compress_probability=0.5,
#                                               jpeg_quality=90),
#                 'label':
#                     ffcv.fields.IntField()
#             },
#                 num_workers=self.opts.data.loaders.num_workers)
            
#             val_writer.from_indexed_dataset(val_dataset, chunksize=100)
    
    
            # Pass a type for each data field
#             val_writer = DatasetWriter(
#                 self.val_write_path,
#                 {
#                     # Tune options to optimize dataset size, throughput at train-time
#                     "rgb": RGBImageField(max_resolution=256),
#                     "near_ir": NDArrayField(dtype=np.dtype("float32"), shape=(1, 256, 256)),
#                     "altitude": NDArrayField(dtype=np.dtype("float32"), shape=(1, 256, 256)),
#                     "landcover": NDArrayField(dtype=np.dtype("float32"), shape=(1, 256, 256)),
#                     "label": IntField(),
#                 },
#                 num_workers=self.opts.data.loaders.num_workers
#             )

#             val_writer.from_indexed_dataset(val_dataset)

            ffcv_monkey_patches()
        else:

            # data and transforms
            self.train_dataset = GeoLifeCLEF2022Dataset(
                self.data_dir,
                self.opts.data.splits.train,
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform= None,
                target_transform=None,
                opts=self.opts
            )

            self.val_dataset = GeoLifeCLEF2022Dataset(
                self.data_dir,
                self.opts.data.splits.val,
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
                opts=self.opts
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = GeoLifeCLEF2022Dataset(
                self.opts.data_dir,
                self.opts.data.splits.test,
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
                opts=self.opts
            )

        if stage == "predict" or stage is None:
            self.test_dataset = GeoLifeCLEF2022Dataset(
                self.opts.data_dir,
                self.opts.data.splits.test,
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
                opts=self.opts
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
