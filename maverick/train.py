import os
import hydra
import omegaconf
import pytorch_lightning as pl
import torch

from pathlib import Path
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from rich.console import Console

from maverick.data.pl_data_modules import BasePLDataModule
from maverick.models.pl_modules import BasePLModule

import wandb
import time
from epoch_timelogger import EpochTimeLogger
from datetime import timedelta
from pytorch_lightning.callbacks import Timer

torch.set_printoptions(edgeitems=100)
os.environ["WANDB_PROJECT"] = "Maverick_incr"

def train(conf: omegaconf.DictConfig) -> None:
    # fancy logger
    console = Console()
    run = wandb.init()


    print("\n=== FINAL CONFIG USED FOR TRAINING ===\n",
        OmegaConf.to_yaml(conf, resolve=True))

    # reproducibility
    pl.seed_everything(conf.train.seed)
    set_determinism_the_old_way(conf.train.pl_trainer.deterministic)
    conf.train.pl_trainer.deterministic = True

    console.log(f"Starting training for [bold cyan]{conf.train.model_name}[/bold cyan] model")
    if conf.train.pl_trainer.fast_dev_run:
        console.log(f"Debug mode {conf.train.pl_trainer.fast_dev_run}. Forcing debugger configuration")
        # Debuggers don't like GPUs nor multiprocessing
        conf.train.pl_trainer.accelerator = "cpu"

        conf.train.pl_trainer.precision = 32
        conf.data.datamodule.num_workers = {k: 0 for k in conf.data.datamodule.num_workers}
        # Switch wandb to offline mode to prevent online logging
        conf.logging.log = None
        # remove model checkpoint callback
        conf.train.model_checkpoint_callback = None

    # data module declaration
    console.log(f"Instantiating the Data Module")
    pl_data_module: BasePLDataModule = hydra.utils.instantiate(conf.data.datamodule, _recursive_=False)
    # force setup to get labels initialized for the model
    pl_data_module.prepare_data()
    pl_data_module.setup("fit")

    # main module declaration
    console.log(f"Instantiating the Model")

    pl_module: BasePLModule = hydra.utils.instantiate(conf.model.module, _recursive_=False)
    print(pl_module)
    # pl_module = BasePLModule.load_from_checkpoint(conf.evaluation.checkpoint, _recursive_=False, map_location="cuda:0")
    experiment_logger: Optional[WandbLogger] = None
    experiment_path: Optional[Path] = None
    if conf.logging.log:
        console.log(f"Instantiating Wandb Logger")
        experiment_logger = hydra.utils.instantiate(conf.logging.wandb_arg)
        experiment_logger.watch(pl_module, **conf.logging.watch)
        experiment_path = Path(experiment_logger.experiment.dir)
        # Store the YaML config separately into the wandb dir
        yaml_conf: str = OmegaConf.to_yaml(cfg=conf)
        (experiment_path / "hparams.yaml").write_text(yaml_conf)

        # callbacks declaration
    callbacks_store = [RichProgressBar()]
    ## tracking time
    timer_cb = Timer(interval="step")              #   step granularity is enough here
    callbacks_store.append(timer_cb)

    callbacks_store.append(EpochTimeLogger())

    if conf.train.early_stopping_callback is not None:
        early_stopping_callback: EarlyStopping = hydra.utils.instantiate(conf.train.early_stopping_callback)
        callbacks_store.append(early_stopping_callback)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(
            conf.train.model_checkpoint_callback,
            dirpath=experiment_path / "checkpoints",
        )
        callbacks_store.append(model_checkpoint_callback)

    if conf.train.learning_rate_callback is not None and not conf.train.pl_trainer.fast_dev_run:
        lr: LearningRateMonitor = hydra.utils.instantiate(conf.train.learning_rate_callback)
        callbacks_store.append(lr)
    # trainer
    console.log(f"Instantiating the Trainer")
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer, callbacks=callbacks_store, logger=experiment_logger)

    
    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)

    # ---- total training wall-time ---------------------------------
    # Lightning’s Timer tracks the stages **train/validate/test/...**
    train_sec = timer_cb.time_elapsed("train") or 0.0          # training loop
    val_sec   = timer_cb.time_elapsed("validate") or 0.0       # validation loop
    total_fit_sec = train_sec + val_sec                        # “fit” = train + val

    console.log(f"Training finished in {timedelta(seconds=int(total_fit_sec))}")
    run.log({"time/fit_sec": total_fit_sec})

    # ----------------------------------------------------------------
    # define best_model_path *once* so we can use it safely later on
    best_model_path = None
    if isinstance(model_checkpoint_callback, ModelCheckpoint):
        best_model_path = model_checkpoint_callback.best_model_path
        console.log(f"Best model path: {best_model_path}")

    # ---- run the test loop -----------------------------------------
    test_results = None
    if best_model_path and os.path.exists(best_model_path):
        console.log(f"Loading best model for testing: {best_model_path}")
        trainer.test(pl_module, datamodule=pl_data_module, ckpt_path=best_model_path)
        total_test_sec = timer_cb.time_elapsed("test") or 0.0
        console.log(f"Test loop took {total_test_sec:.2f}s")
        run.log({"time/test_sec": total_test_sec})
    else:
        console.log("Skipping testing – no valid best-checkpoint was found.")


    # # module test
    # trainer.test(pl_module, datamodule=pl_data_module)
    # Load best model based on the metric monitored by ModelCheckpoint
    best_model_path = model_checkpoint_callback.best_model_path if model_checkpoint_callback else None
    if best_model_path and os.path.exists(best_model_path):
         console.log(f"Loading best model for testing: {best_model_path}")
         test_results = trainer.test(pl_module, datamodule=pl_data_module, ckpt_path=best_model_path)
         console.log("Test results:", test_results)
         # Log test results to the same wandb run
         if test_results and isinstance(test_results, list) and len(test_results) > 0:
              # Prefix test metrics to distinguish them, e.g., "test/f1"
              test_metrics = {"test/" + k: v for k, v in test_results[0].items()}
              run.log(test_metrics)
    else:
         console.log("Skipping testing or best model checkpoint not found.")

    # Finish the wandb run
    run.finish()


def set_determinism_the_old_way(deterministic: bool):
    # determinism for cudnn
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # fixing non-deterministic part of horovod
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
        os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)


# @hydra.main(config_path="./conf", config_name="root", version_base="1.1")
@hydra.main(config_path="./conf", config_name="root", version_base="1.1")
def main(cfg: DictConfig) -> None:
    #print(OmegaConf.to_yaml(cfg, resolve=True))  
    train(cfg)

if __name__ == "__main__":
    main()
