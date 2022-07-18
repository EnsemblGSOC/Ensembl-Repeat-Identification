# standard library
import math
import os
import sys
from typing import Iterable
from tqdm import tqdm

# third party
import numpy as np
import torch
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.utilities import AttributeDict
import yaml

# project
from dataloader import build_dataloader
from model import build_model
from mAP_validation import mean_average_precision


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer,
    num_classes,
    iou_threshold,
    max_norm: float = 0,
):
    model.train()
    criterion.train()
    losssum = 0
    mAPsum = 0
    sample_num = 0
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        sample_num += 1
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        losssum += losses.item()
        mAPsum += mean_average_precision(outputs, targets, num_classes, iou_threshold)
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
    writer.add_scalar("mAP", mAPsum / sample_num, epoch)
    writer.add_scalar("Loss/train", losssum / sample_num, epoch)


def test_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    writer,
):
    losssum = 0
    sample_num = 0
    model.eval()
    criterion.eval()
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            sample_num += 1
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )
            losssum += losses.item()

    writer.add_scalar("Loss/test", losssum / sample_num, epoch)


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

    train_loader, validation_loader = build_dataloader(configuration)

    model, criterion = build_model(configuration)
    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=configuration.lr)

    writer = SummaryWriter()
    for epoch in tqdm(range(configuration.epochs)):
        train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            writer,
            num_classes=configuration.num_classes,
            iou_threshold=configuration.iou_threshold,
        )
        test_one_epoch(
            model,
            criterion,
            validation_loader,
            device,
            epoch,
            writer,
        )


if __name__ == "__main__":
    main()
