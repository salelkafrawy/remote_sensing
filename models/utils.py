import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)


def get_nb_bands(bands):
    """
    Get number of channels in the satellite input branch
    (stack bands of satellite + environmental variables)
    """
    n = 0
    for b in bands:
        if b in ["near_ir", "landcover", "altitude"]:
            n += 1
        elif b == "ped":
            n += 8
        elif b == "bioclim":
            n += 19
        elif b == "rgb":
            n += 3
    return n


def get_scheduler(optimizer, opts):
    if opts.scheduler.name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            factor=opts.scheduler.reduce_lr_plateau.factor,
            patience=opts.scheduler.reduce_lr_plateau.lr_schedule_patience,
        )
    elif opts.scheduler.name == "StepLR":
        return StepLR(
            optimizer, opts.scheduler.step_lr.step_size, opts.scheduler.step_lr.gamma
        )
    elif opts.scheduler.name == "CosineRestarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=opts.scheduler.cosine.t_0,
            T_mult=opts.scheduler.cosine.t_mult,
            eta_min=opts.scheduler.cosine.eta_min,
            last_epoch=opts.scheduler.cosine.last_epoch,
        )
    elif opts.scheduler.name == "OneCycleLR":
        return OneCycleLR(
                optimizer,
                max_lr=opts.scheduler.one_cycle.max_lr,
                total_steps=opts.scheduler.one_cycle.total_steps)
    elif opts.scheduler.name is None:
        return None

    else:
        raise ValueError(f"Scheduler'{opts.scheduler.name}' is not valid")


def get_optimizer(trainable_parameters, opts):
    learning_rate = opts.module.lr
    if opts.optimizer == "Adam":
        optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    elif opts.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    elif opts.optimizer == "SGD":
        optimizer = torch.optim.SGD(trainable_parameters, lr=learning_rate)
    elif opts.optimizer == "SGD+Nesterov":
        optimizer = torch.optim.SGD(
            trainable_parameters,
            nesterov=opts.nestrov,
            momentum=opts.momentum,
            dampening=opts.dampening,
            lr=learning_rate,
        )
    else:
        raise ValueError(f"Optimizer'{opts.optimizer}' is not valid")
    return optimizer


############# custom callbacks ##################
def to_numpy(x):
    return x.cpu().detach().numpy()


class InputMonitorSSL(pl.Callback):
    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):

        if (batch_idx + 1) % trainer.log_every_n_steps == 0:

            # log inputs and targets
            query, [k0, k1, k2] = batch

            # in other models
            #             patches, target, meta = batch
            #             input_patches = patches["input"]
#             assert input_patches.device.type == "cuda"
            #             assert target.device.type == "cuda"


            logger = trainer.logger
            logger.experiment.log_histogram_3d(
                to_numpy(query), "query", step=trainer.global_step
            )
            logger.experiment.log_histogram_3d(
                to_numpy(k0), "k0", step=trainer.global_step
            )
            logger.experiment.log_histogram_3d(
                to_numpy(k1), "k1", step=trainer.global_step
            )
            logger.experiment.log_histogram_3d(
                to_numpy(k2), "k2", step=trainer.global_step
            )
#                         logger.experiment.log_histogram_3d(
#                             to_numpy(target), "target", step=trainer.global_step
#                         )

            # log weights
#             actual_model = next(iter(trainer.model.children()))
#             for name, param in actual_model.named_parameters():
#                 logger.experiment.log_histogram_3d(
#                     to_numpy(param), name=name, step=trainer.global_step
#                 )



class InputMonitorBaseline(pl.Callback):
    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:

            # log inputs and targets
            patches, target, meta = batch
            input_patches = patches["input"]
            assert input_patches.device.type == "cuda"
            assert target.device.type == "cuda"

            logger = trainer.logger
            logger.experiment.log_histogram_3d(
                to_numpy(input_patches), "input", step=trainer.global_step
            )
            
            # log weights
            actual_model = next(iter(trainer.model.children()))
            for name, param in actual_model.named_parameters():
                logger.experiment.log_histogram_3d(
                    to_numpy(param), name=name, step=trainer.global_step
                )


class DeviceCallback(pl.Callback):
    def on_batch_start(self, trainer, pl_module):
        assert next(pl_module.parameters()).device.type == "cuda"
