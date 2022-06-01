from random import shuffle
from torch.utils.data import DataLoader, Dataset
from typing import Union
import pathlib
from os import listdir
from os.path import isfile, join
import numpy as np


class RepeatSequenceDataset(Dataset):
    def __init__(
        self,
        seq_folder: Union[str, pathlib.Path],
        label_folder: Union[str, pathlib.Path],
    ):
        super().__init__()
        self.seqs = self.read_seq(seq_folder)
        self.labels = self.read_labels(label_folder)

    def read_seqs(self, seq_folder: Union[str, pathlib.Path]):
        seq_results = []
        for seq_file in listdir(seq_folder):
            if isfile(join(seq_folder, seq_file)) and seq_file.endswith(".csv"):
                seq_results.append(self.read_seq_file(join(seq_folder, seq_file)))

        # concatenate all chr content
        seq_arrays = seq_results[0]
        for i in range(1, len(seq_results)):
            seq_arrays = np.concatenate((seq_arrays, seq_results[i]), axis=0)
        return seq_arrays

    def read_seq_file(self, seq_filename: Union[str, pathlib.Path]):
        return np.loadtxt(seq_filename, delimiter=",")

    def read_labels(self, label_folder: Union[str, pathlib.Path]):
        pass

    def read_label_file(self, label_filename: Union[str, pathlib.Path]):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def onehot_encode():
    pass


def build_dataloader():
    dataset = RepeatSequenceDataset(
        seq_folder="./ref_datasets/datasets", label_folder="./annotation_label"
    )
    return DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=onehot_encode)
