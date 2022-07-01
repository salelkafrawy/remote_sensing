from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from .ssl_pytorch_dataset import GeoLifeCLEF2022Dataset
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from kornia import image_to_tensor


class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: Tensor = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        return x_out.float() / 255.0
    

class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
            

class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
            
            
class GeoLifeDataModule(pl.LightningDataModule):
    def __init__(self, opts, **kwargs: Any):
        super().__init__()
        self.opts = opts
        self.data_dir = self.opts.data_dir

    def setup(self, stage: Optional[str] = None):

        # data and transforms
        self.train_dataset = GeoLifeCLEF2022Dataset(
            self.data_dir,
            region="both",
            patch_data=self.opts.data.bands,
            use_rasters=False,
            patch_extractor=None,
            transform= Preprocess(),
            target_transform=None,
        )

    def train_dataloader(self):

        train_loader = InfiniteDataLoader(
                self.train_dataset,
                batch_size=self.opts.data.loaders.batch_size,
                num_workers=self.opts.data.loaders.num_workers,
                shuffle=True,
                pin_memory=True,
                drop_last=True
            )
        return train_loader
