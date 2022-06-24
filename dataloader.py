# standard library
import pathlib

from typing import Union


# third party
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from pyfaidx import Fasta
from torch.utils.data import DataLoader, Dataset

# project
from config import repeat_class_IDs


class RepeatSequenceDataset(Dataset):
    def __init__(
        self,
        fasta_path: Union[str, pathlib.Path],
        annotations_path: Union[str, pathlib.Path],
        segment_length: int = 2000,
        overlap: int = 500,
        transform=None,
    ):
        super().__init__()
        self.genome = Fasta(fasta_path)["chr1"]
        self.annotations = pd.read_csv(
            annotations_path, sep="\t", names=["start", "end", "subtype"]
        )
        self.len = len(self.genome)
        self.transform = transform
        self.segment_length = segment_length
        self.overlap = overlap
        self.repeat_list = self.filter_no_repeat()

    def get_the_corresponding_repeat(self, anno_df, start, end):
        repeats_in_sequence = anno_df.loc[
            (
                (anno_df["start"] >= start)
                & (anno_df["end"] <= end)
                & (anno_df["start"] < anno_df["end"])
            )
            # ----------------------------
            # ^seq_start                  ^seq_end
            #             -----------------------
            #             ^rep_start            ^rep_end
            | (
                (anno_df["start"] < end)
                & (end < anno_df["end"])
                & (anno_df["start"] < anno_df["end"])
            )
            #            ----------------------------
            #            ^seq_start                  ^seq_end
            # -----------------------
            # ^rep_start            ^rep_end
            | (
                (anno_df["start"] < start)
                & (start < anno_df["end"])
                & (anno_df["start"] < anno_df["end"])
            )
        ]
        return repeats_in_sequence

    def filter_no_repeat(self):

        repeat_list = []
        for index in tqdm(range(self.len // (self.segment_length - self.overlap))):
            genome_index = index * (self.segment_length - self.overlap)
            anno_df = self.annotations
            start = genome_index
            end = genome_index + self.segment_length
            repeats_in_sequence = self.get_the_corresponding_repeat(anno_df, start, end)
            if not repeats_in_sequence.empty:
                repeats_in_sequence = repeats_in_sequence.apply(
                    lambda x: [
                        max(start, x["start"]),
                        min(end, x["end"]),
                        x["subtype"],
                    ],
                    axis=1,
                    result_type="broadcast",
                )
                repeat_list.append((start, repeats_in_sequence))
        return repeat_list

    def forward_strand(self, index):
        start, repeats_in_sequence = self.repeat_list[index]
        end = start + self.segment_length
        sequence = self.genome[start:end].seq.upper()

        sample = {"sequence": sequence, "start": start}

        repeat_ids_series = repeats_in_sequence["subtype"].map(repeat_class_IDs)
        repeat_ids_array = np.array(repeat_ids_series, np.int32)
        repeat_ids_tensor = torch.tensor(repeat_ids_array, dtype=torch.long)

        coordinates = repeats_in_sequence[["start", "end"]]
        sample, coordinates = self.transform((sample, coordinates))
        target = {"classes": repeat_ids_tensor, "coordinates": coordinates}
        return (sample, target)

    def __getitem__(self, index):
        return self.forward_strand(index)

    def __len__(self):
        return len(self.repeat_list)

    def collate_fn(self, batch):
        sequences = [data[0]["sequence"] for data in batch]
        labels = [data[1] for data in batch]
        return torch.stack(sequences), labels


def build_dataset():
    dataset = RepeatSequenceDataset(
        fasta_path="./data/genome_assemblies/datasets/chr1.fa",
        annotations_path="./data/annotations/hg38_chr1.csv",
        transform=transforms.Compose(
            [
                SampleMapEncode(DnaSequenceMapper()),
                CoordinatesToTensor(),
                NormalizeCoordinates(),
                TranslateCoordinates(),
            ]
        ),
    )
    return dataset


def build_dataloader(args):
    validation_split = args.validation_split
    dataset = build_dataset()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.seed(args.seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=dataset.collate_fn,
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        collate_fn=dataset.collate_fn,
    )
    return train_loader, validation_loader


class SampleMapEncode:
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, item):
        # [n, 2]
        sample, target_df = item
        sample["sequence"] = self.mapper.sequence_to_one_hot(sample["sequence"])
        return (sample, target_df)


class CoordinatesToTensor:
    def __init__(self):
        pass

    def __call__(self, item):
        # [n, 2]
        sample, target_df = item
        target_array = np.array(target_df, dtype=np.float32)
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


class DnaSequenceMapper:
    """
    DNA sequences translation to one-hot or label encoding.
    """

    def __init__(self):
        nucleobase_symbols = ["A", "C", "G", "T", "N"]
        padding_character = [" "]

        self.nucleobase_letters = sorted(nucleobase_symbols + padding_character)

        self.num_nucleobase_letters = len(self.nucleobase_letters)

        self.nucleobase_letter_to_index = {
            nucleobase_letter: index
            for index, nucleobase_letter in enumerate(self.nucleobase_letters)
        }

        self.index_to_nucleobase_letter = {
            index: nucleobase_letter
            for index, nucleobase_letter in enumerate(self.nucleobase_letters)
        }

    def sequence_to_one_hot(self, sequence):
        sequence_indexes = [
            self.nucleobase_letter_to_index[nucleobase_letter]
            for nucleobase_letter in sequence
        ]
        one_hot_sequence = F.one_hot(
            torch.tensor(sequence_indexes), num_classes=self.num_nucleobase_letters
        )
        one_hot_sequence = one_hot_sequence.type(torch.float32)

        return one_hot_sequence

    def sequence_to_label_encoding(self, sequence):
        label_encoded_sequence = [
            self.nucleobase_letter_to_index[nucleobase] for nucleobase in sequence
        ]

        label_encoded_sequence = torch.tensor(label_encoded_sequence, dtype=torch.int32)

        return label_encoded_sequence


if __name__ == "__main__":
    dataset = build_dataset()
    print(len(dataset))
    repeat_dict = dict()
    for repeat in dataset:
        key = repeat[1]["classes"].nelement()
        repeat_dict[key] = repeat_dict.get(key, 0) + 1
    print(repeat_dict)
    # index = 10100
    # index = 0
    # index = 165_970

    # import random

    # print(dataset[0])
    # while True:
    #     index = random.randint(1, 5000)
    #     item = dataset[index]
    #     print(f"{index=}, {item=}")
    #     annotation = item[1]
    #     if annotation["classes"].nelement() > 0:
    #         break
    # # dataloader, _ = build_dataloader()
    # for data in dataloader:
    #     print(data)
