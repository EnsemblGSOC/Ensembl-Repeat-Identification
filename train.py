import math
import os
import sys
from typing import Iterable
import numpy as np
import torch
import random
from dataloader import build_dataloader
from model import build_model


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    model.train()
    criterion.train()

    for samples, targets in data_loader:
        samples = samples.to(device)
        # print(samples.shape)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # if targets["classes"].nelement() <= 0:
        #     continue

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        # print(losses.item()) will be alternative into Tensorboard
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    device = torch.device(device)
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model()
    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    dataloader = build_dataloader()

    for epoch in range(2):
        train_one_epoch(model, criterion, dataloader, optimizer, device, epoch)


if __name__ == "__main__":
    main()
