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


# project
from dataloader import build_dataloader
from model import build_model


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer,
    max_norm: float = 0,
):
    model.train()
    criterion.train()
    losssum = 0
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

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
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
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--validation_split", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    return args


def main():
    args = argument()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    device = torch.device(device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model()
    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader, validation_loader = build_dataloader(args)
    writer = SummaryWriter()
    for epoch in tqdm(range(args.epochs)):
        train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, writer
        )
        test_one_epoch(model, criterion, validation_loader, device, epoch, writer)


if __name__ == "__main__":
    main()
