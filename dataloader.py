# standard library
import pathlib
import pickle

from typing import List, Union

# third party
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from pyfaidx import Fasta
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

# project
from config import repeat_class_IDs
from utils import data_directory


class RepeatSequenceDataset(Dataset):
    def __init__(
        self,
        fasta_path: Union[str, pathlib.Path],
        annotations_path: Union[str, pathlib.Path],
        chromosomes: List[str],
        segment_length: int = 2000,
        overlap: int = 500,
        transform=None,
    ):
        super().__init__()
        self.chromosomes = chromosomes
        self.path = [f"{fasta_path}/{chromosome}.fa" for chromosome in self.chromosomes]
        self.annotation = [
            f"{annotations_path}/hg38_{chromosome}.csv"
            for chromosome in self.chromosomes
        ]

        self.transform = transform
        self.segment_length = segment_length
        self.overlap = overlap
        self.repeat_list = self.select_chr()

    def select_chr(self):
        repeat_list = []
        for fasta_path, chromosome, annotation_path in zip(
            self.path, self.chromosomes, self.annotation
        ):
            annotation_path = pathlib.Path(annotation_path)
            segments_repeats_pickle_path = (
                data_directory / annotation_path.name.replace(".csv", ".pickle")
            )

            # load the segments_with_repeats list from disk if it has already been generated
            if segments_repeats_pickle_path.is_file():
                with open(segments_repeats_pickle_path, "rb") as pickle_file:
                    segments_with_repeats = pickle.load(pickle_file)
            else:
                genome = Fasta(fasta_path)[chromosome]
                annotations = pd.read_csv(
                    annotation_path, sep="\t", names=["start", "end", "subtype"]
                )
                segments_with_repeats = self.get_segments_with_repeats(
                    genome, annotations
                )

                # save the segments_with_repeats list as a pickle file
                with open(segments_repeats_pickle_path, "wb") as pickle_file:
                    pickle.dump(segments_with_repeats, pickle_file)

            repeat_list.extend(segments_with_repeats)

        return repeat_list

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

    def get_segments_with_repeats(self, genome, annotations):
        repeat_list = []
        for index in tqdm(range(len(genome) // (self.segment_length - self.overlap))):
            genome_index = index * (self.segment_length - self.overlap)
            anno_df = annotations
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
                repeat_list.append(
                    (genome[start:end].seq.upper(), start, repeats_in_sequence)
                )
        return repeat_list

    def forward_strand(self, index):
        sequence, start, repeats_in_sequence = self.repeat_list[index]
        end = start + self.segment_length

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


def build_dataloader(configuration):
    validation_split = configuration.validation_split
    dataset = RepeatSequenceDataset(
        fasta_path="./data/genome_assemblies/datasets",
        annotations_path="./data/annotations",
        chromosomes=configuration.chromosomes,
        segment_length=configuration.segment_length,
        overlap=configuration.overlap,
        transform=transforms.Compose(
            [
                SampleMapEncode(DnaSequenceMapper()),
                CoordinatesToTensor(),
                NormalizeCoordinates(),
                TranslateCoordinates(),
            ]
        ),
    )

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.seed(configuration.seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=configuration.batch_size,
        sampler=train_sampler,
        collate_fn=dataset.collate_fn,
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=configuration.batch_size,
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
        sample["sequence"] = self.mapper.sequence_to_label_encoding(sample["sequence"])
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

        self.nucleobase_letters = sorted(nucleobase_symbols)

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
    dataset = RepeatSequenceDataset(
        fasta_path="./data/genome_assemblies/datasets",
        annotations_path="./data/annotations",
        chromosomes=["chrX"],
        transform=transforms.Compose(
            [
                SampleMapEncode(DnaSequenceMapper()),
                CoordinatesToTensor(),
                NormalizeCoordinates(),
                TranslateCoordinates(),
            ]
        ),
    )

    print(dataset[0][0]["sequence"].shape, len(dataset))
    # repeat_dict = dict()
    # for repeat in dataset:
    #     key = repeat[1]["classes"].nelement()
    #     repeat_dict[key] = repeat_dict.get(key, 0) + 1
    # print(repeat_dict)
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
