from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from .pytorch_dataset import GeoLifeCLEF2022Dataset
import transforms.transforms as trf

# from data_loading.ffcv_loader.dataset_ffcv import GeoLifeCLEF2022DatasetFFCV
# from ffcv.writer import DatasetWriter
# from ffcv.fields import RGBImageField, IntField, NDArrayField
# from ffcv.fields.decoders import (
#     IntDecoder,
#     NDArrayDecoder,
#     SimpleRGBImageDecoder,
#     CenterCropRGBImageDecoder,
# )
# from ffcv.loader import Loader, OrderOption
# from ffcv.transforms import (
#     RandomHorizontalFlip,
#     Cutout,
#     NormalizeImage,
#     RandomTranslate,
#     Convert,
#     ToDevice,
#     ToTensor,
#     ToTorchImage,
#     ImageMixup,
# )


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
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
            )

            self.train_write_path = os.path.join(
                self.opts.log_dir, "geolife_train_data.beton"
            )
            # Pass a type for each data field
            writer = DatasetWriter(
                write_path,
                {
                    # Tune options to optimize dataset size, throughput at train-time
                    "rgb": RGBImageField(max_resolution=256),
                    "near_ir": NDArrayField(
                        dtype=np.dtype("float32"), shape=(1, 256, 256)
                    ),
                    "label": IntField(),
                },
            )

            # Write dataset
            writer.from_indexed_dataset(train_dataset)

            val_dataset = GeoLifeCLEF2022DatasetFFCV(
                self.data_dir,
                self.opts.data.splits.val,
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None,
                target_transform=None,
            )

            self.val_write_path = os.path.join(
                self.opts.log_dir, "geolife_val_data.beton"
            )
            # Pass a type for each data field
            writer = DatasetWriter(
                write_path,
                {
                    # Tune options to optimize dataset size, throughput at train-time
                    "rgb": RGBImageField(max_resolution=256),
                    "near_ir": NDArrayField(
                        dtype=np.dtype("float32"), shape=(1, 256, 256)
                    ),
                    "label": IntField(),
                },
            )
            writer.from_indexed_dataset(val_dataset)

        else:

            # data and transforms
            self.train_dataset = GeoLifeCLEF2022Dataset(
                self.data_dir,
                self.opts.data.splits.train,
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform= None, # trf.get_transforms(self.opts, "train"),
                target_transform=None,
                load_envvars=self.opts.module.multimodal,
                opts=self.opts
            )

            self.val_dataset = GeoLifeCLEF2022Dataset(
                self.data_dir,
                self.opts.data.splits.val,
                region="both",
                patch_data=self.opts.data.bands,
                use_rasters=False,
                patch_extractor=None,
                transform=None, #trf.get_transforms(self.opts, "val"),
                target_transform=None,
                load_envvars=self.opts.module.multimodal,
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
                transform=trf.get_transforms(
                    self.opts, "val"
                ),  # transforms.ToTensor(),
                target_transform=None,
                load_envvars=self.opts.module.multimodal,
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
                transform=trf.get_transforms(
                    self.opts, "val"
                ),  # transforms.ToTensor(),
                target_transform=None,
                load_envvars=self.opts.module.multimodal,
                opts=self.opts
            )

    def train_dataloader(self):

        if self.opts.use_ffcv_loader:

            # Data decoding and augmentation (the first one is the left-most)
            img_pipeline = [
                CenterCropRGBImageDecoder(output_size=(224, 224), ratio=0.5),
                RandomHorizontalFlip(flip_prob=0.5),
                ImageMixup(alpha=0.5, same_lambda=True),
                ToTensor(),
                #                 ToDevice(self.device, non_blocking=True),
                ToTorchImage(),
                NormalizeImage(
                    np.array([106.9413, 114.8729, 104.5280]),
                    np.array([51.0005, 44.8594, 43.2014]),
                    np.float16,
                ),
            ]

            input_pipeline = [
                NDArrayDecoder(),
                ToTensor(),
                #                 ToDevice(self.device, non_blocking=True),
                transforms.Normalize([131.0458], [53.0884]),
            ]

            label_pipeline = [
                IntDecoder(),
                ToTensor(),
            ]
            #                 ToDevice(self.device, non_blocking=True)]

            # Pipeline for each data field
            pipelines = {
                "rgb": img_pipeline,
                "near_ir": input_pipeline,
                "label": label_pipeline,
            }

            train_loader = Loader(
                self.train_write_path,
                batch_size=self.opts.data.loaders.batch_size,
                num_workers=self.opts.data.loaders.num_workers,
                order=OrderOption.RANDOM,
                pipelines=pipelines,
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
            # Data decoding and augmentation (the first one is the left-most)
            img_pipeline = [
                CenterCropRGBImageDecoder(output_size=(224, 224), ratio=0.5),
                ImageMixup(alpha=0.5, same_lambda=True),
                ToTensor(),
                #                 ToDevice(0, non_blocking=True),
                ToTorchImage(),
                NormalizeImage(
                    np.array([106.9413, 114.8729, 104.5280]),
                    np.array([51.0005, 44.8594, 43.2014]),
                    np.float16,
                ),
            ]

            input_pipeline = [
                NDArrayDecoder(),
                ToTensor(),
                #                 ToDevice(0, non_blocking=True),
                transforms.Normalize([131.0458], [53.0884]),
            ]

            label_pipeline = [
                IntDecoder(),
                ToTensor(),
            ]
            #                 ToDevice(0, non_blocking=True)]

            # Pipeline for each data field
            pipelines = {
                "rgb": img_pipeline,
                "near_ir": input_pipeline,
                "label": label_pipeline,
            }

            val_loader = Loader(
                self.val_write_path,
                batch_size=self.opts.data.loaders.batch_size,
                num_workers=self.opts.data.loaders.num_workers,
                order=OrderOption.RANDOM,
                pipelines=pipelines,
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
