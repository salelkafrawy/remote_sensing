import numpy as np
import torch
cuda_ver = torch.version.cuda.replace(".", "")
import time

import composer
from composer.models import ComposerResNetCIFAR
from torchvision import datasets, transforms

torch.manual_seed(42) # For replicability


from composer.datasets.ffcv_utils import ffcv_monkey_patches
from composer.datasets.ffcv_utils import write_ffcv_dataset

ffcv_monkey_patches()

device = "gpu"
batch_size = 32
num_workers = 1


import os
import sys
import inspect

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, CURR_DIR)

from dataset.ffcv_loader.dataset_ffcv import GeoLifeCLEF2022DatasetFFCV

save_dir = "/home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/tmp_geo"

import ffcv
import torch
from ffcv.fields.decoders import (
    IntDecoder,
    NDArrayDecoder,
    SimpleRGBImageDecoder,
    CenterCropRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    RandomHorizontalFlip,
    Cutout,
    NormalizeImage,
    RandomTranslate,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
    ImageMixup,
)


ffcv_train_dataset = GeoLifeCLEF2022DatasetFFCV(
    "/network/scratch/s/sara.ebrahim-elkafrawy",
    "train",
    region="both",
    patch_data="all", # self.opts.data.bands,
    use_rasters=False,
    patch_extractor=None,
    transform=None,
    target_transform=None,
    )

train_write_path = os.path.join(
       save_dir , "geolife_train_data.ffcv"
    )

ffcv_val_dataset = GeoLifeCLEF2022DatasetFFCV(
        "/network/scratch/s/sara.ebrahim-elkafrawy",
        "val",
        region="both",
        patch_data="all", #self.opts.data.bands,
        use_rasters=False,
        patch_extractor=None,
        transform=None,
        target_transform=None,
    )

val_write_path = os.path.join(
        save_dir, "geolife_val_data.ffcv"
    )


write_ffcv_dataset(dataset=ffcv_val_dataset, write_path=save_dir + "/geo_val.ffcv")

label_pipeline = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze()]
image_pipeline = [SimpleRGBImageDecoder(), ffcv.transforms.ToTensor(),
                ffcv.transforms.ToTorchImage(channels_last=False, convert_back_int16=False),
                ffcv.transforms.Convert(torch.float32),
                transforms.Normalize(
                    np.array([106.9413, 114.8729, 104.5280]),
                    np.array([51.0005, 44.8594, 43.2014]),),
            ]

ffcv_train_dataloader = ffcv.Loader(
                save_dir + "/geo_train.ffcv",
                batch_size=batch_size,
                num_workers=num_workers,
                order=ffcv.loader.OrderOption.RANDOM,
                pipelines={
                    'image': image_pipeline,
                    'label': label_pipeline
                },
                drop_last=True,
            )
ffcv_val_dataloader = ffcv.Loader(
                save_dir + "/geo_val.ffcv",
                batch_size=batch_size,
                num_workers=num_workers,
                order=ffcv.loader.OrderOption.RANDOM,
                pipelines={
                    'image': image_pipeline,
                    'label': label_pipeline
                },
                drop_last=False,
            )

model = ComposerResNetCIFAR(model_name='resnet_20', num_classes=17037)

optimizer = composer.optim.DecoupledSGDW(
    model.parameters(), # Model parameters to update
    lr=0.05, # Peak learning rate
    momentum=0.9,
    weight_decay=2.0e-3 # If this looks large, it's because its not scaled by the LR as in non-decoupled weight decay
)

train_epochs = "2ep"
trainer = composer.trainer.Trainer(
    model=model,
    train_dataloader=ffcv_val_dataloader,
    eval_dataloader=ffcv_val_dataloader,
    max_duration=train_epochs,
    optimizers=optimizer,
#     schedulers=lr_scheduler,
    device=device,
)

start_time = time.perf_counter()
trainer.fit()
end_time = time.perf_counter()
print(f"It took {end_time - start_time:0.4f} seconds to train")