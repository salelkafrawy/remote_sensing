import comet_ml
import os
import sys
from pathlib import Path

from os.path import expandvars

import hydra
from addict import Dict
from omegaconf import OmegaConf, DictConfig

from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from typing import Any, Dict, Tuple, Type, cast
import pdb

from data_loading.pytorch_dataset import GeoLifeCLEF2022Dataset
from trainer.trainer import CNNBaseline


@hydra.main(config_path="configs", config_name="hydra")
def main(opts):

    # prepare configurations from hydra and experiment config file
    opts_dct = dict(OmegaConf.to_container(opts))

    hydra_args = opts_dct.pop("args", None)

    exp_config_name = hydra_args["config_file"]
    machine_abs_path = Path(__file__).resolve().parents[3]
    exp_config_path = machine_abs_path / "configs" / exp_config_name
    trainer_config_path = machine_abs_path / "configs" / "trainer.yaml"

    # fetch the requiered arguments
    exp_opts = OmegaConf.load(exp_config_path)
    trainer_opts = OmegaConf.load(trainer_config_path)
    all_opts = OmegaConf.merge(exp_opts, hydra_args)
    all_opts = OmegaConf.merge(all_opts, trainer_opts)
    exp_configs = cast(DictConfig, all_opts)
    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(exp_configs.trainer))

    # set the seed
    pl.seed_everything(exp_configs.seed)

    # check if the save path exists
    # save experiment in a sub-dir with the config_file name (e.g. save_path/cnn_baseline)
    exp_save_path = os.path.join(
        exp_configs.save_path, exp_configs.config_file.split(".")[0]
    )
    if not os.path.exists(exp_save_path):
        os.makedirs(exp_save_path)

    # save the experiment configurations in the save path
    with open(os.path.join(exp_save_path, "exp_configs.yaml"), "w") as fp:
        OmegaConf.save(config=exp_configs, f=fp)

    ################################################
    # setup comet logging
    if exp_configs.log_comet:

        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            save_dir=exp_save_path,  # Optional
            project_name=exp_configs.comet.project_name,
        )
        comet_logger.experiment.add_tags(list(exp_configs.comet.tags))
        print(exp_configs.comet.tags)
        trainer_args["logger"] = comet_logger

    #     tb_logger = pl_loggers.comet(exp_configs.log_dir, name=exp_config_name)
    ################################################

    # define the callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=exp_save_path,
        save_top_k=3,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=4, mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer_args["callbacks"] = [
        checkpoint_callback,
        lr_monitor,
    ]  # , #early_stopping_callback]
    trainer_args["max_epochs"] = 1000
    # trainer_args["profiler"]="simple"
    trainer_args["overfit_batches"] = 5
    # trainer_args["track_grad_norm"]=2

    batch_size = exp_configs.data.loaders.batch_size
    num_workers = exp_configs.data.loaders.num_workers

    train_dataset = GeoLifeCLEF2022Dataset(
        exp_configs.dataset_path,
        exp_configs.data.splits.train,  # "train+val"
        region="both",
        patch_data="rgb",
        use_rasters=False,
        patch_extractor=None,
        transform=transforms.ToTensor(),
        target_transform=None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    val_dataset = GeoLifeCLEF2022Dataset(
        exp_configs.dataset_path,
        exp_configs.data.splits.val,
        region="both",
        patch_data="rgb",
        use_rasters=False,
        patch_extractor=None,
        transform=transforms.ToTensor(),
        target_transform=None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    model = CNNBaseline(exp_configs)

    trainer = pl.Trainer(max_epochs=1, gpus=1)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,)
               #deterministic=True, )


if __name__ == "__main__":
    main()
