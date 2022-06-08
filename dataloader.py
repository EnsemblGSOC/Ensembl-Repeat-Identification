from random import shuffle
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
        self.genome = Fasta(fasta_name)
        self.annotations = pd.read_csv(
            label_folder, sep="\t", names=["start", "end", "subtype"]
        )
        self.len = len(self.genome)

    def __getitem__(self, index):
        genome_index = index * 1500
        start = genome_index
        end = genome_index + 2000
        sample = self.genome[start:end].seq

        target = self.annotations.apply(
            lambda x: start <= x["start"] and x["end"] <= end, axis=1
        )
        return (sample, target)

    def __len__(self):
        return self.len // 1500


def build_dataloader():
    dataset = RepeatSequenceDataset(
        fasta_name="./data/genome_assemblies/datasets/chr1.fa",
        label_folder="./data/annotations",
    )
    return DataLoader(dataset, batch_size=64, shuffle=True)


if __name__ == "__main__":
    dataloader = build_dataloader()
    print(dataloader[0])
