import math
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)



class MocoLRSchedulerV2(pl.Callback):
    def __init__(self, initial_lr=0.03, use_cosine_scheduler=False, schedule=(120, 160), max_epochs=200):
        super().__init__()
        self.lr = initial_lr
        self.use_cosine_scheduler = use_cosine_scheduler
        self.schedule = schedule
        self.max_epochs = max_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        lr = self.lr

        if self.use_cosine_scheduler:  # cosine lr schedule
            lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / self.max_epochs))
        else:  # stepwise lr schedule
            for milestone in self.schedule:
                lr *= 0.5 if epoch >= milestone else 1.0

        optimizer = trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
def zero_aware_normalize(embedding, axis):
    normalized = torch.nn.functional.normalize(embedding, p=2, dim=axis)
    norms = torch.norm(embedding, p=2, dim=axis, keepdim=True)
    is_zero_norm = (norms == 0).expand_as(normalized)
    return torch.where(is_zero_norm, torch.zeros_like(embedding), normalized)


class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False, decay=1
    ):
        super().__init__(
            optimizer,
            T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
            verbose=verbose,
        )
        self.decay = decay
        self.initial_lrs = self.base_lrs

    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print(f"multiplying base_lrs by {self.decay}")
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
            else:
                n = 0

            self.base_lrs = [
                initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs
            ]

        super().step(epoch)


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


def get_scheduler(optimizer, opts, train_set_length):
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
        epochs = opts.scheduler.cosine.epochs
        bs = opts.data.loaders.batch_size
        steps_per_epoch = train_set_length // bs
        t_0 = steps_per_epoch * epochs
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0,
            T_mult=opts.scheduler.cosine.t_mult,
            eta_min=opts.scheduler.cosine.eta_min,
            last_epoch=opts.scheduler.cosine.last_epoch,
        )
    elif opts.scheduler.name == "CosineResDecay":
        epochs = opts.scheduler.cosine_decay.epochs
        bs = opts.data.loaders.batch_size
        steps_per_epoch = train_set_length // bs
        t_0 = steps_per_epoch * epochs
        return CosineAnnealingWarmRestartsDecay(
            optimizer,
            T_0=t_0,
            T_mult=opts.scheduler.cosine_decay.t_mult,
            eta_min=opts.scheduler.cosine_decay.eta_min,
            last_epoch=opts.scheduler.cosine_decay.last_epoch,
            decay=opts.scheduler.cosine_decay.decay,
        )
    elif opts.scheduler.name == "OneCycleLR":
        return OneCycleLR(
            optimizer,
            max_lr=opts.scheduler.one_cycle.max_lr,
            epochs=opts.max_epochs,
            steps_per_epoch=train_set_length // opts.data.loaders.batch_size,
        )
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
