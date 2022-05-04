import comet_ml
import os
import sys
from pathlib import Path

from os.path import expandvars

import hydra
from addict import Dict
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
from typing import Any, Dict, Tuple, Type, cast
import pdb

import transforms.transforms as trf
from data_loading.pytorch_dataset import GeoLifeCLEF2022Dataset
from trainer.trainer import CNNBaseline, CNNMultitask


@hydra.main(config_path="configs", config_name="hydra")
def main(opts):

    # prepare configurations from hydra and experiment config file
    opts_dct = dict(OmegaConf.to_container(opts))

    hydra_args = opts_dct.pop("args", None)

    exp_config_name = hydra_args["config_file"]

    current_file_path = hydra.utils.to_absolute_path(__file__)
    machine_abs_path = Path(current_file_path).parent
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
    exp_configs.save_path = exp_save_path
    exp_configs.preds_file = os.path.join(
        exp_configs.save_path,
        exp_configs.comet.experiment_name + "_predictions.csv",
    )
    # save the experiment configurations in the save path
    with open(os.path.join(exp_save_path, "exp_configs.yaml"), "w") as fp:
        OmegaConf.save(config=exp_configs, f=fp)

    ################################################
    # define the callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=exp_save_path,
        save_top_k=3,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00001, patience=10, mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer_args["callbacks"] = [
        checkpoint_callback,
        lr_monitor,
        #         early_stopping_callback,
    ]

    batch_size = exp_configs.data.loaders.batch_size
    num_workers = exp_configs.data.loaders.num_workers

    model = CNNBaseline(exp_configs)  # CNNMultitask(exp_configs)

    trainer = pl.Trainer(
        max_epochs=trainer_args["max_epochs"],
        gpus=1,
        #         logger=comet_logger,
        log_every_n_steps=trainer_args["log_every_n_steps"],
        callbacks=trainer_args["callbacks"],
        track_grad_norm=2,
        detect_anomaly=True,
        overfit_batches=trainer_args[
            "overfit_batches"
        ],  ## make sure it is 0.0 when training
    )

    trainer.test(
        model, ckpt_path="/network/scratch/t/tengmeli/ecosystem_project/exps/multigpu_baseline/last.ckpt"
    )  # or ckpt path (e.g. '/home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/cnn_baseline/last.ckpt')

#        ckpt_path="/home/mila/s/sara.ebrahim-elkafrawy/scratch/ecosystem_project/exps/rgb_reduceLROnPlateau_nestrov/last.ckpt",
 #   )


if __name__ == "__main__":
    main()
