# standard library
import math
import os
import sys
from typing import Iterable
from tqdm import tqdm
import datetime as dt
import warnings
import pathlib
import logging

# third party
import numpy as np
import torch
import random
import argparse

from pytorch_lightning.utilities import AttributeDict
import yaml
import pytorch_lightning as pl

# project
from dataloader import build_dataloader, build_seq2seq_dataset
from model import DETR, build_criterion
from seq2seq import Seq2SeqTransformer
from transformer import Transformer
from utils import logger, logging_formatter_time_message


def argument():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument(
        "--configuration", type=str, help="experiment configuration file path"
    )
    parser.add_argument(
        "--datetime",
        help="datetime string; if set this will be used instead of generating a new one",
    )
    args = parser.parse_args()
    return args


def main():
    args = argument()
    warnings.filterwarnings(
        "ignore",
        ".*does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument.*",
    )
    with open(args.configuration) as file:
        configuration = yaml.safe_load(file)
    configuration = AttributeDict(configuration)

    if args.datetime:
        configuration.datetime = args.datetime
    else:
        configuration.datetime = dt.datetime.now().isoformat(
            sep="_", timespec="seconds"
        )

    experiment_name = f"{configuration.experiment_prefix}_{configuration.dataset_id}_{configuration.datetime}"

    experiments_directory = configuration.save_directory

    experiment_directory = pathlib.Path(f"{experiments_directory}/{experiment_name}")

    experiment_directory.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(f"{experiment_directory}/test_output.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging_formatter_time_message)
    logger.addHandler(file_handler)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    device = torch.device(device)
    seed = configuration.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    training_dataloader, validation_dataloader, test_dataloader = build_seq2seq_dataset(
        configuration
    )
    model = Seq2SeqTransformer(
        num_encoder_layers=configuration.num_encoder_layers,
        num_decoder_layers=configuration.num_decoder_layers,
        emb_size=configuration.embedding_dimension,
        nhead=configuration.nhead,
        src_vocab_size=configuration.num_nucleobase_letters,
        tgt_vocab_size=configuration.num_nucleobase_letters
        + configuration.num_classes
        + 2,
        configuration=configuration,
    )
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
        callbacks=[early_stopping_callback],
        log_every_n_steps=1,
        profiler=configuration.profiler,
    )

    trainer.fit(
        model=model,
        train_dataloaders=training_dataloader,
        val_dataloaders=validation_dataloader,
    )
    trainer.test(ckpt_path="best", dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
