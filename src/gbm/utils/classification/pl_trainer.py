import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger

from gbm.utils.pipeline.config import Config


def get_classification_trainer(
        config: Config,
        **kwargs
) -> pl.Trainer:
    """
    Returns the PyTorch Lightning trainer for a classification problem.

    Args:
        config (Config):
            A dictionary containing the configuration parameters for the trainer.
        **kwargs:
            Additional keyword arguments.

    Returns:
        pl.Trainer:
            The PyTorch Lightning trainer for the classification problem.
    """

    # Defining callbacks
    callbacks = [
        # Defining early stopping callback
        pl.callbacks.EarlyStopping(
            monitor="validation_loss",
            min_delta=0.001,
            patience=5
        ),
        # Defining checkpointing callback
        pl.callbacks.ModelCheckpoint(
            filename="{epoch}-{training_loss:.2f}-{validation_loss:.2f}",
            monitor="validation_loss"
        ),
        # Defining learning rate monitor callback
        pl.callbacks.LearningRateMonitor(
            logging_interval="step"
        )
    ]

    # Defining loggers
    loggers = [
        TensorBoardLogger(
            save_dir=config.get("path_to_logs"),
            name="tensorboard_logs"
        ),
        CSVLogger(
            save_dir=config.get("path_to_logs"),
            name="csv_logs"
        )
        # pl.loggers.WandbLogger(
        #    save_dir=config.get("path_to_logs"),
        #    name="wandb_logs"
        # )
    ]

    # Defining trainer settings
    lightning_trainer = pl.Trainer(
        max_epochs=config.get("num_epochs"),
        val_check_interval=1/config.get("num_checks_per_epoch"),
        accumulate_grad_batches=config.get("gradient_accumulation_steps"),
        callbacks=callbacks,
        accelerator=config.get("device"),
        logger=loggers
    )

    return lightning_trainer
