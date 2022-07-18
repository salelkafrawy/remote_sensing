import comet_ml

import os
import sys
import pdb
import timeit

from pathlib import Path
from typing import Any, Dict, Tuple, Type, cast


import hydra
from omegaconf import OmegaConf, DictConfig

from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler
from pytorch_lightning.profiler.pytorch import PyTorchProfiler

from models.seco_resnets import SeCoCNN
from models.cnn_finetune import CNNBaseline
from models.mosaiks import MOSAIKS
from models.multitask import CNNMultitask
from models.multimodal_envvars import MultimodalTabular

from dataset.geolife_datamodule import GeoLifeDataModule
from composer.datasets.ffcv_utils import ffcv_monkey_patches
from composer.datasets.ffcv_utils import write_ffcv_dataset

from models.utils import InputMonitor

from types import MethodType
from dataset.ffcv_loader import custom_PTL_methods 
from dataset.ffcv_loader.dataset_ffcv import GeoLifeCLEF2022DatasetFFCV
from dataset.ffcv_loader.utils import FFCV_PIPELINES
from ffcv.loader import Loader, OrderOption


def get_ffcv_dataloaders(exp_configs):
    
    train_dataset = GeoLifeCLEF2022DatasetFFCV(
    exp_configs.data_dir,
    exp_configs.data.splits.train,
    region="both",
    patch_data="all", # self.opts.data.bands,
    use_rasters=False,
    patch_extractor=None,
    transform=None,
    target_transform=None,
    )

    train_write_path = os.path.join(
        exp_configs.log_dir, "geolife_train_data.ffcv"
    )
    write_ffcv_dataset(dataset=train_dataset, write_path=train_write_path)

    val_dataset = GeoLifeCLEF2022DatasetFFCV(
        exp_configs.data_dir,
        exp_configs.data.splits.val,
        region="both",
        patch_data="all", #self.opts.data.bands,
        use_rasters=False,
        patch_extractor=None,
        transform=None,
        target_transform=None,
    )

    val_write_path = os.path.join(
        exp_configs.log_dir, "geolife_val_data.ffcv"
    )
    write_ffcv_dataset(dataset=val_dataset, write_path=val_write_path)

    ffcv_monkey_patches()

    train_loader = Loader(
                train_write_path,
                batch_size=exp_configs.data.loaders.batch_size,
                num_workers=exp_configs.data.loaders.num_workers,
                order=OrderOption.RANDOM,
                pipelines=FFCV_PIPELINES,
            )
    val_loader = Loader(
                val_write_path,
                batch_size=exp_configs.data.loaders.batch_size,
                num_workers=exp_configs.data.loaders.num_workers,
                order=OrderOption.RANDOM,
                pipelines=FFCV_PIPELINES,
            )
    return train_loader, val_loader
    

@hydra.main(config_path="configs", config_name="hydra")
def main(opts):

    # prepare configurations from hydra and experiment config file
    opts_dct = dict(OmegaConf.to_container(opts))

    hydra_args = opts_dct.pop("args", None)
    data_dir = opts_dct.pop("data_dir", None)
    log_dir = opts_dct.pop("log_dir", None)

    mosaiks_weights_path = opts_dct.pop("mosaiks_weights_path", None)
    random_init_path = opts_dct.pop("random_init_path", None)

    current_file_path = hydra.utils.to_absolute_path(__file__)

    exp_config_name = hydra_args["config_file"]
    machine_abs_path = Path(current_file_path).parent
    exp_config_path = machine_abs_path / "configs" / exp_config_name
    trainer_config_path = machine_abs_path / "configs" / "trainer.yaml"

    # fetch the requiered arguments
    exp_opts = OmegaConf.load(exp_config_path)
    trainer_opts = OmegaConf.load(trainer_config_path)

    all_opts = OmegaConf.merge(exp_opts, hydra_args)
    all_opts = OmegaConf.merge(all_opts, trainer_opts)

    all_opts["data_dir"] = data_dir
    all_opts["log_dir"] = log_dir
    all_opts["mosaiks_weights_path"] = mosaiks_weights_path

    all_opts["mosaiks_weights_path"] = mosaiks_weights_path
    all_opts["random_init_path"] = random_init_path
    

    exp_configs = cast(DictConfig, all_opts)
    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(exp_configs.trainer))

    # set the seed
    pl.seed_everything(exp_configs.seed)

    # check if the log dir exists
    if not os.path.exists(exp_configs.log_dir):
        os.makedirs(exp_configs.log_dir)

    # prediction file name
    exp_configs.preds_file = os.path.join(
        exp_configs.log_dir,
        exp_configs.comet.experiment_name + "_predictions.csv",
    )

    # save the experiment configurations in the save path
    with open(os.path.join(exp_configs.log_dir, "exp_configs.yaml"), "w") as fp:
        OmegaConf.save(config=exp_configs, f=fp)

    ################################################

    # setup comet logging
#     if exp_configs.log_comet:

#         comet_logger = CometLogger(
#             api_key=os.environ.get("COMET_API_KEY"),
#             workspace=os.environ.get("COMET_WORKSPACE"),
#             save_dir=exp_configs.log_dir,  # Optional
#             experiment_name=exp_configs.comet.experiment_name,
#             project_name=exp_configs.comet.project_name,
#             #             auto_histogram_gradient_logging=True,
#             #             auto_histogram_activation_logging=True,
#             #             auto_histogram_weight_logging=True,
#             log_code=False,
#         )
#         comet_logger.experiment.add_tags(list(exp_configs.comet.tags))
#         comet_logger.log_hyperparams(exp_configs)
#         trainer_args["logger"] = comet_logger


    ################################################
    # define the callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_topk-error",
        dirpath=exp_configs.log_dir,
        save_top_k=3,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_topk-error", min_delta=0.00001, patience=10, mode="min"

    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer_args["callbacks"] = [
        checkpoint_callback,
        lr_monitor,
        early_stopping_callback,
#         InputMonitor(),
    ]

    # data loaders
    geolife_datamodule = GeoLifeDataModule(exp_configs)

    if exp_configs.task == "base":
        model = CNNBaseline(exp_configs)
        if "seco" in exp_configs.module.model:
            model = SeCoCNN(exp_configs)
        
    elif exp_configs.task == "multi":
        model = CNNMultitask(exp_configs)

    elif exp_configs.task == "multimodal":
        model = MultimodalTabular(exp_configs)
        
    elif exp_configs.task == "mosaiks":
        model = MOSAIKS(exp_configs)

    #     profiler = SimpleProfiler(filename="profiler_simple.txt")
    #     profiler = AdvancedProfiler(filename="profiler_advanced.txt")
    #     profiler = PyTorchProfiler(filename="profiler_pytorch.txt")

    trainer = pl.Trainer(
        enable_progress_bar=True,
        default_root_dir=exp_configs.log_dir,
        max_epochs=exp_configs.max_epochs,
        gpus=exp_configs.gpus,
#         logger=comet_logger,
        log_every_n_steps=trainer_args["log_every_n_steps"],
        callbacks=trainer_args["callbacks"],
        overfit_batches=trainer_args[
            "overfit_batches"
        ],  ## make sure it is 0.0 when training
        precision=16,
        accumulate_grad_batches=int(exp_configs.data.loaders.batch_size / 4),
#         progress_bar_refresh_rate=0,
        #         strategy="ddp_find_unused_parameters_false",
        #         distributed_backend='ddp',
        #         profiler=profiler,
#         weights_summary="full",
#         track_grad_norm=1,
        gradient_clip_val=1.5,
        num_sanity_val_steps=-1
    )

    start = timeit.default_timer()
    
    if exp_configs.use_ffcv_loader:
        ffcv_train_loader, ffcv_val_loader = get_ffcv_dataloaders(exp_configs)
        
#         trainer.fit_loop.epoch_loop.on_run_start = MethodType(custom_PTL_methods.on_run_start, trainer.fit_loop.epoch_loop)
#         trainer.fit_loop.epoch_loop.advance = MethodType(custom_PTL_methods.advance, trainer.fit_loop.epoch_loop)
        
        
        trainer.fit(model, train_dataloaders=ffcv_train_loader, val_dataloaders=ffcv_val_loader)
    else:
        trainer.fit(model, datamodule=geolife_datamodule)
        
    stop = timeit.default_timer()

    print("Elapsed fit time: ", stop - start)


if __name__ == "__main__":
    main()
