from random import shuffle
from turtle import forward
from torch import ge
from torch.utils.data import DataLoader, Dataset
from typing import Union
import pathlib
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyfaidx import Fasta
from config import chr_length


class RepeatSequenceDataset(Dataset):
    def __init__(
        self,
        fasta_name: Union[str, pathlib.Path],
        label_folder: Union[str, pathlib.Path],
    ):
        super().__init__()
        self.genome = Fasta(fasta_name)["chr1"]
        self.annotations = pd.read_csv(
            label_folder, sep="\t", names=["start", "end", "subtype"]
        )
        self.len = len(self.genome)

    def forward_strand(self, index):
        genome_index = index * 1500
        start = genome_index
        end = genome_index + 2000
        sample = self.genome[start:end].seq

        target = self.annotations.apply(
            lambda x: start <= x["start"] and x["end"] <= end and x["start"] < x["end"],
            axis=1,
        )
        target = self.annotations[target]
        return (sample, target)

    def reverse_strand(self, index):
        genome_index = index * 1500
        start = genome_index
        end = genome_index + 2000
        sample = self.genome[start:end].reverse.seq

        target = self.annotations.apply(
            lambda x: start <= x["end"] and x["start"] <= end and x["end"] < x["start"],
            axis=1,
        )
        target = self.annotations[target]
        return (sample, target)

    def __getitem__(self, index):
        if index < (self.len // 1500):
            return self.forward_strand(index)
        else:
            return self.reverse_strand(index - (self.len // 1500))

    def __len__(self):
        return (self.len // 1500) * 2


def build_dataset():
    dataset = RepeatSequenceDataset(
        fasta_name="./data/genome_assemblies/datasets/chr1.fa",
        label_folder="./data/annotations/hg38_chr1.csv",
    )
    return dataset


def build_dataloader():
    return DataLoader(build_dataset(), batch_size=64, shuffle=True)


if __name__ == "__main__":
    dataset = build_dataset()
    print(dataset[10100])
