import comet_ml
import os
import sys
from pathlib import Path
import timeit

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
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler
from pytorch_lightning.profiler.pytorch import PyTorchProfiler
from typing import Any, Dict, Tuple, Type, cast
import pdb

import transforms.transforms as trf
from data_loading.pytorch_dataset import GeoLifeCLEF2022Dataset
from trainer.trainer import CNNBaseline, CNNMultitask

# import git


def to_numpy(x):
    return x.cpu().detach().numpy()


class InputMonitor(pl.Callback):
    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):

        if (batch_idx + 1) % trainer.log_every_n_steps == 0:

            # log inputs and targets
            patches, target, meta = batch
            input_patches = patches["input"]

            logger = trainer.logger
            logger.experiment.log_histogram_3d(
                to_numpy(input_patches), "input", step=trainer.global_step
            )
            logger.experiment.log_histogram_3d(
                to_numpy(target), "target", step=trainer.global_step
            )


#             # log weights
#             actual_model = next(iter(trainer.model.children()))
#             for name, param in actual_model.named_parameters():
#                 logger.experiment.log_histogram_3d(
#                     to_numpy(param), name=name, step=trainer.global_step
#                 )


@hydra.main(config_path="configs", config_name="hydra")
def main(opts):

    # prepare configurations from hydra and experiment config file
    opts_dct = dict(OmegaConf.to_container(opts))

    hydra_args = opts_dct.pop("args", None)

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
    exp_configs = cast(DictConfig, all_opts)
    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(exp_configs.trainer))

    # fetch git repo hash
    #    repo = git.Repo(search_parent_directories=True)
    #    repo_sha = repo.head.object.hexsha

    # set the seed
    pl.seed_everything(exp_configs.seed)

    # check if the save path exists
    # save experiment in a sub-dir with the config_file name (e.g. save_path/cnn_baseline)
    exp_save_path = os.path.join(
        exp_configs.save_path,
        exp_configs.comet.experiment_name,  # exp_configs.config_file.split(".")[0]
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

    # setup comet logging
    if exp_configs.log_comet:

        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            save_dir=exp_save_path,  # Optional
            experiment_name=exp_configs.comet.experiment_name,
            project_name=exp_configs.comet.project_name,
            #             auto_histogram_gradient_logging=True,
            #             auto_histogram_activation_logging=True,
            #             auto_histogram_weight_logging=True,
            log_code=False,
        )
        comet_logger.experiment.add_tags(list(exp_configs.comet.tags))
        comet_logger.log_hyperparams(exp_configs)
        #       comet_logger.log_hyperparams({"git_sha": repo_sha})
        trainer_args["logger"] = comet_logger

    #    comet_logger.experiment.log_code(
    #        filename=hydra.utils.to_absolute_path(__file__)
    #    )

    ################################################
    # define the callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=exp_save_path,
        save_top_k=3,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="topk-error", min_delta=0.00001, patience=10, mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer_args["callbacks"] = [
        checkpoint_callback,
        lr_monitor,
        early_stopping_callback,
        # InputMonitor(),
    ]

    if exp_configs.task == "multi":
        print("Using Multitask")
        model = CNNMultitask(exp_configs) 
    else:
        model = CNNBaseline(exp_configs)  # CNNMultitask(exp_configs)

#     profiler = SimpleProfiler(filename="profiler_simple.txt")
#     profiler = AdvancedProfiler(filename="profiler_advanced.txt")
    profiler = PyTorchProfiler(filename="profiler_pytorch.txt")
   

    trainer = pl.Trainer(
        enable_progress_bar=True,
        default_root_dir=exp_configs.save_path,
        max_epochs=trainer_args["max_epochs"],
        gpus=trainer_args["gpus"],
        logger=comet_logger,
        log_every_n_steps=trainer_args["log_every_n_steps"],
        callbacks=trainer_args["callbacks"],
        strategy="ddp_find_unused_parameters_false",
        #detect_anomaly=True,
        overfit_batches=trainer_args[
            "overfit_batches"
        ],  ## make sure it is 0.0 when training
#         profiler=profiler,
        precision=16,
        accumulate_grad_batches=int(exp_configs.data.loaders.batch_size/4),
        
#         distributed_backend='ddp',
    )


    start = timeit.default_timer()
    trainer.fit(model)
    # for cnn multigpu baseline, ckpt_path = "/network/scratch/t/tengmeli/ecosystem_project/exps/multigpu_baseline/last.ckpt")
    #db.set_trace()
    stop = timeit.default_timer()

    print("Elapsed fit time: ", stop - start)


if __name__ == "__main__":
    main()
