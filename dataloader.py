from torch.utils.data import DataLoader, Dataset
from typing import Union
import pathlib


class RepeatSequenceDataset(Dataset):
    def __init__(
        self,
        seq_folder: Union[str, pathlib.Path],
        label_folder: Union[str, pathlib.Path],
    ):
        super().__init__()
        self.seqs = self.read_seq(seq_folder)
        self.labels = self.read_labels(label_folder)

    def read_seq(self, seq_folder: Union[str, pathlib.Path]):
        pass

    def read_labels(self, label_folder: Union[str, pathlib.Path]):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
