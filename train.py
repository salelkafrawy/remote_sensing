import os
import sys
import pdb
import timeit
from pathlib import Path
from typing import Any, Dict, Tuple, Type, cast
import logging

import comet_ml
import hydra
from omegaconf import OmegaConf, DictConfig

from torchvision import transforms
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

from models.utils import InputMonitorBaseline

from dataset.geolife_datamodule import GeoLifeDataModule

# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

# configure logging on module level, redirect to file
logger = logging.getLogger("pytorch_lightning.core")
logger.addHandler(logging.FileHandler("core.log"))


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
    cnn_ckpt_path = opts_dct.pop("cnn_ckpt_path", None)
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
    all_opts["mosaiks_weights_path"] = mosaiks_weights_path

    all_opts["mosaiks_weights_path"] = mosaiks_weights_path
    all_opts["random_init_path"] = random_init_path
    all_opts["mocov2_ssl_ckpt_path"] = mocov2_ssl_ckpt_path
    all_opts["cnn_ckpt_path"] = cnn_ckpt_path
    all_opts["ffcv_write_path"] = ffcv_write_path

    exp_configs = cast(DictConfig, all_opts)
    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(exp_configs.trainer))

    # set the seed
    pl.seed_everything(exp_configs.seed, workers=True)

    
    if exp_configs.cnn_ckpt_path == "":
        recent_epoch = 0
        ckpt_file_path = None
    else: #make sure the ckpt name ends in '_epochNum.ckpt'
        recent_epoch = int(exp_configs.cnn_ckpt_path.split('_')[-1].split('.')[0])
        ckpt_file_path = exp_configs.cnn_ckpt_path
    
    # check if the log dir exists
    if not os.path.exists(exp_configs.log_dir):
        os.makedirs(exp_configs.log_dir)
    else: # check if there is a current checkpoint
        files = os.listdir(exp_configs.log_dir)
        for f_name in files:
            file_path = os.path.join(exp_configs.log_dir, f_name)
            if os.path.isfile(file_path):
                file_ext = f_name.split('.')[1]

                if file_ext == 'ckpt' and f_name != 'last.ckpt':
                    epoch_num = int(f_name.split('.')[0].split('=')[-1])
                    if epoch_num > recent_epoch:
                        ckpt_file_path = file_path
                        recent_epoch = epoch_num

    logger.info(f'ckpt_file:{ckpt_file_path}')
    

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
#         early_stopping_callback,
#         InputMonitorBaseline(),
    ]

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

    trainer = pl.Trainer(
        enable_progress_bar=True,
        default_root_dir=exp_configs.log_dir,
        max_epochs=exp_configs.max_epochs,
        gpus=exp_configs.gpus,
        logger=comet_logger,
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
#                 track_grad_norm=1,
        gradient_clip_val=1.5,
#         num_sanity_val_steps=-1,
    )

    start = timeit.default_timer()

    geolife_datamodule = GeoLifeDataModule(exp_configs)
    trainer.fit(model, datamodule=geolife_datamodule, ckpt_path=ckpt_file_path,)

    stop = timeit.default_timer()

    print("Elapsed fit time: ", stop - start)


if __name__ == "__main__":
    main()
