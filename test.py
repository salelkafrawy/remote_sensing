import os
import sys
import pdb
import timeit
from pathlib import Path
from typing import Any, Dict, Tuple, Type, cast
import logging

import hydra
from omegaconf import OmegaConf, DictConfig

from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
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

from models.utils import InputMonitorBaseline

from dataset.geolife_datamodule import GeoLifeDataModule


@hydra.main(config_path="configs", config_name="hydra")
def main(opts):

    # prepare configurations from hydra and experiment config file
    opts_dct = dict(OmegaConf.to_container(opts))

    hydra_args = opts_dct.pop("args", None)
    data_dir = opts_dct.pop("data_dir", None)
    log_dir = opts_dct.pop("log_dir", None)
    exp_name = opts_dct.pop("exp_name", None)
    cnn_ckpt_path = opts_dct.pop("cnn_ckpt_path", None)

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
    all_opts["exp_name"] = exp_name
    all_opts["cnn_ckpt_path"] = cnn_ckpt_path
    all_opts["testing"] = True

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
        exp_configs.exp_name + "_predictions.csv",
    )

    ################################################

    # data loaders
    geolife_datamodule = GeoLifeDataModule(exp_configs)

    if exp_configs.task == "base":
        model = CNNBaseline(exp_configs)

    elif exp_configs.task == "seco":
        model = SeCoCNN(exp_configs)

    elif exp_configs.task == "multi":
        model = CNNMultitask(exp_configs)

    elif exp_configs.task == "multimodal":
        model = MultimodalTabular(exp_configs)

    elif exp_configs.task == "mosaiks":
        model = MOSAIKS(exp_configs)
    trainer = pl.Trainer(
        max_epochs=exp_configs.max_epochs,
        gpus=1,
    )

    trainer.test(
        model,
        #         ckpt_path=cnn_ckpt_path,
        datamodule=geolife_datamodule,
    )


if __name__ == "__main__":
    main()
