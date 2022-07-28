import os
import numpy as np
import torch
from torch import nn
from pytorch_lightning import Callback
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from sklearn.metrics import average_precision_score
from torch.utils.data import random_split, DataLoader

from dataset.geolife_datamodule import GeoLifeDataModule
import transforms.transforms as trf
from dataset.pytorch_dataset import GeoLifeCLEF2022Dataset

import ffcv
from ffcv.fields.decoders import (
    IntDecoder,
    NDArrayDecoder,
    SimpleRGBImageDecoder,
    CenterCropRGBImageDecoder,
)
from torchvision import transforms

from ffcv.loader import Loader, OrderOption
from ffcv.fields import RGBImageField, IntField, NDArrayField
from ffcv.writer import DatasetWriter
from composer.datasets.ffcv_utils import write_ffcv_dataset
from composer.datasets.ffcv_utils import ffcv_monkey_patches

# from .utils import FFCV_PIPELINES

def get_data_loaders(opts):
#     train_dataset = GeoLifeCLEF2022Dataset(
#                 opts.data_dir,
#                 opts.data.splits.train,
#                 opts.use_ffcv_loader,
#                 region="both",
#                 patch_data="all",  # self.opts.data.bands,
#                 use_rasters=False,
#                 patch_extractor=None,
#                 transform=None,
#                 target_transform=None,
#             )

#     train_write_path = os.path.join(
#         opts.ffcv_write_path, "geolife_train_data.ffcv"
#     )
#     write_ffcv_dataset(dataset=train_dataset, write_path=train_write_path)

#     val_dataset = GeoLifeCLEF2022Dataset(
#         opts.data_dir,
#         opts.data.splits.val,
#         opts.use_ffcv_loader,
#         region="both",
#         patch_data="all",  # self.opts.data.bands,
#         use_rasters=False,
#         patch_extractor=None,
#         transform=None,
#         target_transform=None,
#     )

#     val_write_path = os.path.join(
#         opts.ffcv_write_path, "geolife_val_data.ffcv"
#     )
#     write_ffcv_dataset(dataset=val_dataset, write_path=val_write_path)

#     ffcv_monkey_patches()
    
    
#     img_pipeline = [
#     CenterCropRGBImageDecoder(output_size=(224, 224), ratio=0.5),
#     ffcv.transforms.RandomHorizontalFlip(flip_prob=0.5),
#     ffcv.transforms.ImageMixup(alpha=0.5, same_lambda=True),
#     ffcv.transforms.ToTensor(),
#     ffcv.transforms.ToTorchImage(channels_last=False, convert_back_int16=False),
#     ffcv.transforms.Convert(torch.float32),
#     transforms.Normalize(
#         np.array([106.9413, 114.8729, 104.5280]),
#         np.array([51.0005, 44.8594, 43.2014]),
#     ),
#     ]


#     lbl_pipeline = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze()]



#     train_loader = Loader(
#                 train_write_path,
#                 batch_size=opts.data.loaders.batch_size,
#                 num_workers=opts.data.loaders.num_workers,
#                 order=OrderOption.RANDOM,
#                 pipelines= {
#                                 "image": img_pipeline,
#                                 "label": lbl_pipeline,
#                             }
# ,
#             )
#     val_loader = Loader(
#                 val_write_path,
#                 batch_size=opts.data.loaders.batch_size,
#                 num_workers=opts.data.loaders.num_workers,
#                 order=OrderOption.RANDOM,
#                 pipelines={
#                                 "image": img_pipeline,
#                                 "label": lbl_pipeline,
#                             },
#             )

    train_dataset = GeoLifeCLEF2022Dataset(
        opts.data_dir,
        opts.data.splits.train,
        False,
        region="both",
        patch_data=opts.data.bands,
        use_rasters=False,
        patch_extractor=None,
        transform=None,
        target_transform=None,
        opts=opts,
    )

    val_dataset = GeoLifeCLEF2022Dataset(
        opts.data_dir,
        opts.data.splits.val,
        False,
        region="both",
        patch_data=opts.data.bands,
        use_rasters=False,
        patch_extractor=None,
        transform=None,
        target_transform=None,
        opts=opts,
    )
    
    train_loader = DataLoader(
                train_dataset,
                batch_size=opts.data.loaders.batch_size,
                num_workers=opts.data.loaders.num_workers,
                shuffle=True,
                pin_memory=True,
            )
    
    val_loader = DataLoader(
                val_dataset,
                batch_size=opts.data.loaders.batch_size,
                num_workers=opts.data.loaders.num_workers,
                shuffle=False,
                pin_memory=True,
            )
    return train_loader, val_loader

            
class SSLOnlineEvaluator(Callback):
    def __init__(self, opts, data_dir, z_dim, batch_size=1024, num_workers=32):
        self.opts = opts
        self.z_dim = z_dim
        self.max_epochs = opts.ssl.online_max_epochs
        self.check_val_every_n_epoch = opts.ssl.online_val_every_n_epoch

#         self.datamodule = GeoLifeDataModule(opts)
#         self.datamodule.setup()

        self.train_loader, self.val_loader = get_data_loaders(opts)
        self.loss = nn.CrossEntropyLoss()

    def on_pretrain_routine_start(self, trainer, pl_module):
        self.classifier = SSLEvaluator(
            n_input=self.z_dim, n_classes=self.opts.num_species, n_hidden=None
        ).to(pl_module.device)

        self.optimizer = torch.optim.SGD(self.classifier.parameters(), lr=0.01)

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.check_val_every_n_epoch != 0:
            return

        encoder = pl_module.encoder_q

        self.classifier.train()
        for _ in range(self.max_epochs):

            for batch in self.train_loader: #self.datamodule.train_dataloader():
                patches, target, meta = batch
                patches["rgb"] = patches["rgb"].to(pl_module.device)
                target = target.to(pl_module.device)
                patches = trf.get_transforms(self.opts, "train")(patches)
                input_patches = patches["rgb"]

                with torch.no_grad():
                    outputs = encoder(input_patches)
                outputs = outputs.detach()

                logits = self.classifier(outputs)
                loss = self.loss(logits, target)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.classifier.eval()
        top_k_vals = []
        for batch in self.val_loader: #self.datamodule.val_dataloader():
            patches, target, meta = batch
            patches["rgb"] = patches["rgb"].to(pl_module.device)
            target = target.to(pl_module.device)
            patches = trf.get_transforms(self.opts, "val")(patches)
            input_patches = patches["rgb"]

            with torch.no_grad():
                outputs = encoder(input_patches)
            outputs = outputs.detach()

            logits = self.classifier(outputs)

            # compute top-30 metric
            probas = torch.nn.functional.softmax(logits, dim=1)
            k = 30
            n_classes = probas.shape[1]
            _, pred_ids = torch.topk(probas, k)
            pointwise_accuracy = torch.sum(pred_ids == target[:, None], axis=1)
            top_k = 1 - torch.Tensor.float(pointwise_accuracy).mean()
            top_k_vals.append(top_k)

        top_k_val_set = torch.mean(torch.tensor(top_k_vals))

        metrics = {"online_val_top-k": top_k_val_set}
        trainer.logger.log_metrics(metrics, {})


#         trainer.logger_connector.add_progress_bar_metrics(metrics)
