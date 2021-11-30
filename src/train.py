from Models.splinedist import SplineDist
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar, ModelSummary
import sys
sys.path.append('.')
from Datasets.DSB18 import Nuclie_datamodule
pl.seed_everything(42)


logger = pl_loggers.TensorBoardLogger(
    save_dir="logs",
    name="spline_dist",
    version=1,
    log_graph=True,
    default_hp_metric=False)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="Models/weights",
    filename="splineDist-{epoch:02d}-{val_loss:.2f}",
    save_top_k=5,
    mode="min",
)

earlystoping_callback = EarlyStopping()

lr_logger = LearningRateMonitor(logging_interval="step")
summary = ModelSummary(max_depth=-1)
progressbar = RichProgressBar()

callbacks = [summary, progressbar, checkpoint_callback,
             lr_logger, earlystoping_callback]


if __name__ == "__main__":
    model = SplineDist()
    datamodule = Nuclie_datamodule()

    trainer = pl.Trainer(max_epochs=50,
                         gpus=0,
                         precision=16,
                         callbacks=callbacks,
                         logger=logger,
                         profiler="simple")
    trainer.fit(model, datamodule)
