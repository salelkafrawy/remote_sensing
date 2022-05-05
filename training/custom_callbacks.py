import pytorch_lightning as pl


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
            assert input_patches.device.type == "cuda"
            #             assert patches.device.type == "cuda"
            assert target.device.type == "cuda"


#             assert meta.device.type == "cuda"
#             logger = trainer.logger
#             logger.experiment.log_histogram_3d(
#                 to_numpy(input_patches), "input", step=trainer.global_step
#             )
#             logger.experiment.log_histogram_3d(
#                 to_numpy(target), "target", step=trainer.global_step
#             )


#             # log weights
#             actual_model = next(iter(trainer.model.children()))
#             for name, param in actual_model.named_parameters():
#                 logger.experiment.log_histogram_3d(
#                     to_numpy(param), name=name, step=trainer.global_step
#                 )


class DeviceCallback(pl.Callback):
    def on_batch_start(self, trainer, pl_module):
        assert next(pl_module.parameters()).device.type == "cuda"
