import comet_ml

import os
import sys
import pdb
import timeit
import numpy as np

from pathlib import Path
from typing import Any, Dict, Tuple, Type, cast

import torch
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

from models.moco2_module import MocoV2
from pl_bolts.models.self_supervised.moco.callbacks import MocoLRScheduler

from dataset.ssl.ssl_geolife_datamodule import GeoLifeDataModule
from dataset.ssl.ssl_pytorch_dataset import GeoLifeCLEF2022DatasetSSL

from models.ssl_online import SSLOnlineEvaluator
from models.utils import InputMonitorSSL


@hydra.main(config_path="configs", config_name="hydra")
def main(opts):

    # prepare configurations from hydra and experiment config file
    opts_dct = dict(OmegaConf.to_container(opts))

    hydra_args = opts_dct.pop("args", None)
    data_dir = opts_dct.pop("data_dir", None)
    log_dir = opts_dct.pop("log_dir", None)
    mosaiks_weights_path = opts_dct.pop("mosaiks_weights_path", None)
    random_init_path = opts_dct.pop("random_init_path", None)
    mocov2_ssl_ckpt_path = opts_dct.pop("mocov2_ssl_ckpt_path", None)
    ffcv_write_path = opts_dct.pop("ffcv_write_path", None)

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
    all_opts["random_init_path"] = random_init_path

    all_opts["mosaiks_weights_path"] = mosaiks_weights_path
    all_opts["random_init_path"] = random_init_path
    all_opts["mocov2_ssl_ckpt_path"] = mocov2_ssl_ckpt_path
    all_opts["ffcv_write_path"] = ffcv_write_path

    exp_configs = cast(DictConfig, all_opts)
    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(exp_configs.trainer))

    # set the seed
    pl.seed_everything(exp_configs.seed, workers=True)
    
    if exp_configs.mocov2_ssl_ckpt_path == "":
        recent_epoch = 0
        ckpt_file_path = None
    else:
        recent_epoch = int(exp_configs.mocov2_ssl_ckpt_path.split('=')[1].split('.')[0])
        ckpt_file_path = exp_configs.mocov2_ssl_ckpt_path
    
    # check if the log dir exists
    if not os.path.exists(exp_configs.log_dir):
        os.makedirs(exp_configs.log_dir)
    else: # check if there is a current checkpoint
        files = os.listdir(exp_configs.log_dir)
        for f_name in files:
            file_path = os.path.join(exp_configs.log_dir, f_name)
            if os.path.isfile(file_path):
                file_ext = f_name.split('.')[1]

                if file_ext == 'ckpt':
                    epoch_num = int(f_name.split('.')[0].split('=')[1])
                    if epoch_num > recent_epoch:
                        ckpt_file_path = file_path
                        recent_epoch = epoch_num

    print(f'ckpt_file:{ckpt_file_path}')
    
    # prediction file name
    exp_configs.preds_file = os.path.join(
        exp_configs.log_dir,
        exp_configs.comet.experiment_name + "_predictions.csv",
    )

    # save the experiment configurations in the save path
    with open(os.path.join(exp_configs.log_dir, "exp_configs.yaml"), "w") as fp:
        OmegaConf.save(config=exp_configs, f=fp)

    
    ################################################

#     setup comet logging
    if exp_configs.log_comet:

        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            save_dir=exp_configs.log_dir,  # Optional
            experiment_name=exp_configs.comet.experiment_name,
            project_name=exp_configs.comet.project_name,
            #             auto_histogram_gradient_logging=True,
            #             auto_histogram_activation_logging=True,
            #             auto_histogram_weight_logging=True,
            log_code=False,
        )
        comet_logger.experiment.add_tags(list(exp_configs.comet.tags))
        comet_logger.log_hyperparams(exp_configs)
        trainer_args["logger"] = comet_logger

        comet_logger.experiment.set_code(
            filename=hydra.utils.to_absolute_path(__file__)
        )

    ################################################
    if exp_configs.task == "ssl":
        model = MocoV2(exp_configs)

    # define the callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_configs.log_dir, filename="{epoch}"
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    moco_scheduler = MocoLRScheduler(
        initial_lr=exp_configs.ssl.learning_rate,
        schedule=exp_configs.ssl.schedule,
        max_epochs=exp_configs.max_epochs,
    )

    online_evaluator = SSLOnlineEvaluator(
        exp_configs,
        data_dir=exp_configs.data_dir,
        z_dim=model.mlp_dim,
    )

    trainer_args["callbacks"] = [
        checkpoint_callback,
        lr_monitor,
        moco_scheduler,
#         online_evaluator,
#         InputMonitorSSL()
    ]

    trainer = pl.Trainer(
        enable_progress_bar=True,
        default_root_dir=exp_configs.log_dir,
        max_epochs=exp_configs.max_epochs,
        gpus=exp_configs.gpus,
        accelerator=exp_configs.ssl.accelerator,
        #         devices=exp_configs.ssl.devices,
        #         num_nodes=exp_configs.ssl.num_nodes,
        strategy=exp_configs.ssl.strategy,
        logger=comet_logger,
        log_every_n_steps=trainer_args["log_every_n_steps"],
        callbacks=trainer_args["callbacks"],
        overfit_batches=trainer_args[
            "overfit_batches"
        ],  ## make sure it is 0.0 when training
        precision=16,
        accumulate_grad_batches=int(exp_configs.data.loaders.batch_size / 4),
#         track_grad_norm=1,
        #         profiler=profiler,
    )

    start = timeit.default_timer()

    geolife_datamodule = GeoLifeDataModule(exp_configs)
    trainer.fit(model, datamodule=geolife_datamodule , ckpt_path=ckpt_file_path,)

    stop = timeit.default_timer()

    print("Elapsed fit time: ", stop - start)


if __name__ == "__main__":
    main()

#     parameters_to_vector(model.parameters())
#     hashlib.sha224(b"Nobody inspects the spammish repetition").hexdigest()
# 'a4337bc45a8fc544c03f52dc550cd6e1e87021bc896588bd79e901e2'
