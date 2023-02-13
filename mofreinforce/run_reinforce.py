import copy
import os

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from reinforce.config_reinforce import ex

from predictor.config_predictor import config as predictor_config
from predictor.config_predictor import _loss_names
from predictor.module import Predictor

from generator.config_generator import config as generator_config
from generator.datamodule import GeneratorDatamodule
from generator.module import Generator

from reinforce.module import Reinforce

@ex.automain
def main(_config):
    # 1. load predictor
    predictors = []
    for i in range(len(_config["predictor_load_path"])):
        pred_config = predictor_config()
        pred_config["test_only"] = True
        pred_config["loss_names"] = _loss_names({"regression": 1})
        pred_config["load_path"] = _config["predictor_load_path"][i]
        pred_config["mean"] = _config["mean"][i]
        pred_config["std"] = _config["std"][i]

        predictor = Predictor(pred_config)
        predictor.eval()
        predictors.append(predictor)

    # 2. load generator
    gen_config = generator_config()
    gen_config["load_path"] = _config["generator_load_path"]
    gen_config["batch_size"] = _config["batch_size"]
    gen_config["dataset_dir"] = _config["dataset_dir"]

    generator = Generator(gen_config)
    dm = GeneratorDatamodule(gen_config)

    # 3. set reinforce
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    model = Reinforce(generator, predictors, _config)

    exp_name = f"{_config['exp_name']}"
    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/total_reward",
        mode="max",
        save_last=True,
    )

    logger = pl.loggers.TensorBoardLogger(

        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=_config["devices"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        deterministic=True,
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=_config["max_epochs"],
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=_config["accumulate_grad_batches"],
        log_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        limit_train_batches=_config["limit_train_batches"],
        limit_val_batches=_config["limit_val_batches"],
        num_sanity_val_steps=_config["limit_val_batches"],
        val_check_interval=_config["val_check_interval"],
        gradient_clip_val=_config["gradient_clip_val"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)


