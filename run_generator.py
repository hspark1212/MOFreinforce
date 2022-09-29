import copy
import os

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from generator.config_generator import ex
from generator.datamodule import GeneratorDatamodule
from generator.module import Generator


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = GeneratorDatamodule(_config)

    model = Generator(_config)

    exp_name = f"{_config['exp_name']}"

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )

    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_devices = _config["num_devices"]
    if isinstance(num_devices, list):
        num_devices = len(num_devices)

    # gradient accumulation
    accumulate_grad_batches = _config["batch_size"] // (
            _config["per_gpu_batchsize"] * num_devices * _config["num_nodes"]
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_devices,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        deterministic=True,
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=_config["max_epochs"],
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        val_check_interval=_config["val_check_interval"],
        gradient_clip_val=_config["gradient_clip_val"],
        # profiler="simple",
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
        # trainer.test(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)


