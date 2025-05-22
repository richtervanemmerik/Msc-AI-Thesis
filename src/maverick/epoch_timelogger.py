import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger


class EpochTimeLogger(Callback):
    """
    Prints the current epoch number and how long the epoch took (mm:ss).
    Also pushes the value to wandb if a WandbLogger is attached.
    """
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        # keep a private timer for this epoch
        self._epoch_start_time = time.time()

        current_epoch = trainer.current_epoch + 1      # 1‑based for humans
        max_epochs    = trainer.max_epochs or "?"
        print(f"\n=== Epoch {current_epoch}/{max_epochs} ===")

    def on_train_epoch_end(self, trainer, pl_module, unused: None = None) -> None:
        dur = time.time() - self._epoch_start_time
        mins, secs = divmod(dur, 60)
        print(f"Epoch {trainer.current_epoch + 1} finished in {int(mins):02d}:{int(secs):02d}")

        # optional: log to wandb so it shows up in the run dashboard
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log(
                {"time/epoch_sec": dur, "epoch": trainer.current_epoch + 1}
            )
