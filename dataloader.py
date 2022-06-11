from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from typing import Union
import pathlib
import pandas as pd
from pyfaidx import Fasta
import torch
import numpy as np
import csv
from config import classID


class RepeatSequenceDataset(Dataset):
    def __init__(
        self,
        fasta_name: Union[str, pathlib.Path],
        label_folder: Union[str, pathlib.Path],
        transform=None,
    ):
        super().__init__()
        self.genome = Fasta(fasta_name)["chr1"]
        self.annotations = pd.read_csv(
            label_folder, sep="\t", names=["start", "end", "subtype"]
        )
        self.len = len(self.genome)
        self.transform = transform

    def forward_strand(self, index):
        genome_index = index * 1500
        # produce sequence with overlap 1500 of length 2000
        start = genome_index
        end = genome_index + 2000
        seq = self.genome[start:end].seq

        sample = {"seq": seq, "ID": start}

        target = self.annotations.apply(
            lambda x: start <= x["start"] and x["end"] <= end and x["start"] < x["end"],
            axis=1,
        )
        target = self.annotations[target]
        target["subtype"] = target["subtype"].apply(lambda ty: classID[ty])
        temp_array = np.array(target["subtype"], np.int32)
        subtype = torch.tensor(temp_array, dtype=torch.int32)
        if self.transform:
            # print(target[:, :2])
            pos = target.iloc[:, [0, 1]]
            sample, pos = self.transform((sample, pos))
        target = {"class": subtype, "pos": pos}
        return (sample, target)

    def __getitem__(self, index):
        return self.forward_strand(index)

    def __len__(self):
        return self.len // 1500


def build_dataset():
    dataset = RepeatSequenceDataset(
        fasta_name="./data/genome_assemblies/datasets/chr1.fa",
        label_folder="./data/annotations/hg38_chr1.csv",
        transform=transforms.Compose(
            [TransFormat(), NormalizeLabels(), CenterLength()]
        ),
    )
    return dataset


def build_dataloader():
    return DataLoader(
        build_dataset(),
        batch_size=64,
        shuffle=True,
    )


class TransFormat(object):
    def __init__(
        self,
    ):
        pass

    def __call__(self, data):
        # [n, 2]
        sample, target_pd = data
        temp_array = np.array(target_pd)
        target = torch.tensor(temp_array, dtype=torch.float32)
        return (sample, target)


class NormalizeLabels(object):
    """Normalize the datasets."""

    def __init__(
        self,
    ):
        pass

    def __call__(self, data):
        sample, target = data
        length = len(sample["seq"])
        start_pos = sample["ID"]
        target[:, :] -= start_pos
        target[:, :] /= length
        return (sample, target)


class CenterLength(object):
    """convert (left, right) to (center, length)"""

    def __init__(
        self,
    ):
        pass

    def __call__(self, data):
        # [n, 2]
        sample, target = data
        center = (target[:, 1] + target[:, 0]) / 2  # [n]
        length = (target[:, 1]) - target[:, 0]  # [n]

        return (sample, torch.stack((center, length), axis=1))


if __name__ == "__main__":
    dataset = build_dataset()

    print(dataset[10100])
