# standard library
import math
import os
import sys
from typing import Iterable
from tqdm import tqdm
import datetime as dt

# third party
import numpy as np
import torch
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.utilities import AttributeDict
import yaml
import pytorch_lightning as pl

# project
from dataloader import build_dataloader
from model import DETR, build_criterion
from transformer import build_transformer


def argument():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument(
        "--configuration", type=str, help="experiment configuration file path"
    )
    args = parser.parse_args()
    return args


def main():
    args = argument()
    with open(args.configuration) as file:
        configuration = yaml.safe_load(file)
    configuration = AttributeDict(configuration)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    device = torch.device(device)
    seed = configuration.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    training_dataloader, validation_dataloader = build_dataloader(configuration)

    transformer = build_transformer(configuration)

    criterion = build_criterion(configuration)
    model = DETR(
        transformer,
        num_classes=configuration.num_classes,
        num_queries=configuration.num_queries,
        criterion=criterion,
        configuration=configuration,
    )

    configuration.datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")

    configuration.logging_version = f"{configuration.experiment_prefix}_{configuration.dataset_id}_{configuration.datetime}"

    tensorboard_logger = pl.loggers.TensorBoardLogger(
        save_dir=configuration.save_directory,
        name="",
        version=configuration.logging_version,
        default_hp_metric=False,
    )

    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="validation_loss",
        min_delta=configuration.loss_delta,
        patience=configuration.patience,
        verbose=True,
    )

    trainer = pl.Trainer(
        gpus=configuration.gpus,
        logger=tensorboard_logger,
        max_epochs=configuration.max_epochs,
        log_every_n_steps=1,
        callbacks=[early_stopping_callback],
        profiler=configuration.profiler,
    )

    trainer.fit(
        model=model,
        train_dataloaders=training_dataloader,
        val_dataloaders=validation_dataloader,
    )


if __name__ == "__main__":
    main()
