# standard library
import pathlib

from typing import Union

# third party
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from pyfaidx import Fasta
from torch.utils.data import DataLoader, Dataset

# project
from config import repeat_class_IDs


class RepeatSequenceDataset(Dataset):
    def __init__(
        self,
        fasta_path: Union[str, pathlib.Path],
        annotations_path: Union[str, pathlib.Path],
        transform=None,
    ):
        super().__init__()
        self.genome = Fasta(fasta_path)["chr1"]
        self.annotations = pd.read_csv(
            annotations_path, sep="\t", names=["start", "end", "subtype"]
        )
        self.len = len(self.genome)
        self.transform = transform

    def forward_strand(self, index):
        genome_index = index * 1500
        # produce sequence with overlap 1500 of length 2000
        start = genome_index
        end = genome_index + 2000
        sequence = self.genome[start:end].seq

        sample = {"sequence": sequence, "start": start}

        target = self.annotations.apply(
            lambda x: start <= x["start"] and x["end"] <= end and x["start"] < x["end"],
            axis=1,
        )
        target = self.annotations[target]
        target["subtype"] = target["subtype"].apply(lambda ty: repeat_class_IDs[ty])
        temp_array = np.array(target["subtype"], np.int32)
        subtypes = torch.tensor(temp_array, dtype=torch.int32)
        if self.transform:
            # print(target[:, :2])
            coordinates = target.iloc[:, [0, 1]]
            sample, coordinates = self.transform((sample, coordinates))
        target = {"classes": subtypes, "coordinates": coordinates}
        return (sample, target)

    def __getitem__(self, index):
        return self.forward_strand(index)

    def __len__(self):
        return self.len // 1500


def build_dataset():
    dataset = RepeatSequenceDataset(
        fasta_path="./data/genome_assemblies/datasets/chr1.fa",
        annotations_path="./data/annotations/hg38_chr1.csv",
        transform=transforms.Compose(
            [
                CoordinatesToTensor(),
                NormalizeCoordinates(),
                TranslateCoordinates(),
            ]
        ),
    )
    return dataset


def build_dataloader():
    dataset = build_dataset()
    return DataLoader(dataset, batch_size=64, shuffle=True)


class CoordinatesToTensor:
    def __init__(self):
        pass

    def __call__(self, item):
        # [n, 2]
        sample, target_df = item
        target_array = np.array(target_df)
        target_tensor = torch.tensor(target_array, dtype=torch.float32)
        return (sample, target_tensor)


class NormalizeCoordinates:
    """Normalize a sample's repeat annotation coordinates to a relative location
    in the sequence, defined as start and end floats between 0 and 1."""

    def __init__(self):
        pass

    def __call__(self, item):
        sample, coordinates = item
        length = len(sample["sequence"])
        start_coordinate = sample["start"]
        coordinates[:, :] -= start_coordinate
        coordinates[:, :] /= length
        return (sample, coordinates)


class TranslateCoordinates:
    """Convert (start, end) relative coordinates to (center, span)."""

    def __init__(self):
        pass

    def __call__(self, item):
        # [n, 2]
        sample, target = item
        center = (target[:, 1] + target[:, 0]) / 2  # [n]
        span = (target[:, 1]) - target[:, 0]  # [n]

        return (sample, torch.stack((center, span), axis=1))


if __name__ == "__main__":
    dataset = build_dataset()

    # index = 10100
    # index = 0
    # index = 165_970

    import random

    while True:
        index = random.randint(1, 165_970)
        item = dataset[index]
        print(f"{index=}, {item=}")
        annotation = item[1]
        if annotation["classes"].nelement() > 0:
            break
