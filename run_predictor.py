import copy
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from predictor.config_predictor import ex
from predictor.datamodule import Datamodule
from predictor.module import Predictor


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    exp_name = f"{_config['exp_name']}"

    dm = Datamodule(_config)
    model = Predictor(_config)

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_last=True
    )

    logger = TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )
    callbacks = [checkpoint_callback]

    num_gpus = _config["num_gpus"]
    if isinstance(num_gpus, list):
        num_gpus = len(num_gpus)

    # gradient accumulation
    accumulate_grad_batches = _config["batch_size"] // (
            _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        strategy="ddp",
        benchmark=True,
        max_epochs=_config["max_epochs"],
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main()
