from random import shuffle
from torch.utils.data import DataLoader, Dataset
from typing import Union
import pathlib
from os import listdir
from os.path import isfile, join
import numpy as np
from tqdm import tqdm
from pyfaidx import Fasta
from config import chr_length


class RepeatSequenceDataset(Dataset):
    def __init__(
        self,
        seq_folder: Union[str, pathlib.Path],
        label_folder: Union[str, pathlib.Path],
    ):
        super().__init__()
        self.targets = self.read_FASTA_file(seq_folder, label_folder)

    def read_FASTA_file(
        self,
        seq_filename: Union[str, pathlib.Path],
        label_folder: Union[str, pathlib.Path],
    ):
        print("generate fasta sequence")
        targets = []
        genes = Fasta(seq_filename)
        labels = self.read_labels(label_folder)
        for chr, len in chr_length.items():
            for i in tqdm(range(0, len, 1500)):
                sample = genes[chr][i : i + 2000].seq
                label = self.find_corresponding_label(i, i + 2000, labels[chr])
                targets.append((sample, label))
        return targets

    def find_corresponding_label(self, seq_start: int, seq_end: int, labels):
        results = []
        for label in labels:
            align_start, align_end = int(label[0]), int(label[1])
            if align_start > align_end:
                continue
            if seq_start <= align_start and align_end <= seq_end:
                results.append(label)
        return np.stack(results) if len(results) > 0 else []

    def read_label_file(self, label_filename: Union[str, pathlib.Path]):
        return np.loadtxt(label_filename, delimiter="\t", dtype=str)

    def read_labels(self, label_folder: Union[str, pathlib.Path]):
        dic = dict()
        for chr, len in chr_length.items():
            dic[chr] = self.read_label_file(join(label_folder, f"hg38_{chr}.csv"))
        return dic

    def __getitem__(self, index):
        return self.targets[index]

    def __len__(self):
        return len(self.targets)


def onehot_encode():
    pass


def build_dataloader():
    dataset = RepeatSequenceDataset(
        seq_folder="./data/genome_assemblies/hg38.fa", label_folder="./data/annotations"
    )
    return DataLoader(dataset, batch_size=64, shuffle=True)


if __name__ == "__main__":
    dataloader = build_dataloader()
    print(dataloader[0])
