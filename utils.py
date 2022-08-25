# standard library
import gzip
import hashlib
import logging
import pathlib
import shutil
import sys

from typing import List, Type, Union

# third party
import pandas as pd
import requests
import torch

from torchvision import transforms
from tqdm import tqdm


# logging formats
logging_formatter_time_message = logging.Formatter(
    fmt="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging_formatter_message = logging.Formatter(fmt="%(message)s")

# set up base logger
logger = logging.getLogger("main_logger")
logger.setLevel(logging.DEBUG)
logger.propagate = False
# create console handler and add to logger
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging_formatter_time_message)
logger.addHandler(console_handler)

# set and create dataset directories
data_directory = pathlib.Path("data")
data_directory.mkdir(exist_ok=True)
genome_assemblies_directory = data_directory / "genome_assemblies"
genome_assemblies_directory.mkdir(exist_ok=True)
annotations_directory = data_directory / "annotations"
annotations_directory.mkdir(exist_ok=True)

hits_column_dtypes = {
    "seq_name": "string",
    "family_acc": "string",
    "family_name": "string",
    "bits": "float",
    "e-value": "float",
    "bias": "float",
    "hmm-st": "int",
    "hmm-en": "int",
    "strand": "string",
    "ali-st": "int",
    "ali-en": "int",
    "env-st": "int",
    "env-en": "int",
    "sq-len": "int",
    "kimura_div": "float",
}


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

    def label_encoding_to_sequence(self, label_encoded_sequence):
        sequence = [
            self.index_to_nucleobase_letter[label] for label in label_encoded_sequence
        ]
        return "".join(sequence)

    def label_encoding_to_nucleobase_letter(self, label):
        if label in self.index_to_nucleobase_letter.keys():
            return self.index_to_nucleobase_letter[label]
        else:
            return label


class RepeatSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        genome_fasta_path: Union[str, pathlib.Path],
        annotation_path: Union[str, pathlib.Path],
        chromosomes: List[str],
        dna_sequence_mapper: Type[DnaSequenceMapper],
        segment_length: int = 2000,
        overlap: int = 500,
        transform=None,
    ):
        super().__init__()

        self.chromosomes = chromosomes
        self.dna_sequence_mapper = dna_sequence_mapper
        self.path = [
            f"{genome_fasta_path}/{chromosome}.fa" for chromosome in self.chromosomes
        ]
        self.annotation = [
            f"{annotations_path}/hg38_{chromosome}.csv"
            for chromosome in self.chromosomes
        ]

        self.segment_length = segment_length
        self.overlap = overlap
        self.transform = transform
        self.repeat_list = self.select_chr()

        repeat_types = self.get_unique_categories()
        print(len(repeat_types), repeat_types)
        self.repeat_type_mapper = CategoryMapper(repeat_types)

    def get_unique_categories(self):
        df_list = [
            pd.read_csv(annotation_path, sep="\t", names=["start", "end", "subtype"])
            for annotation_path in self.annotation
        ]
        return sorted(pd.concat(df_list)["subtype"].unique().tolist())

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
        # end = start + self.segment_length

        sample = {"sequence": sequence, "start": start}

        repeat_ids_series = repeats_in_sequence["subtype"].map(
            self.repeat_type_mapper.label_to_index
        )
        repeat_ids_array = np.array(repeat_ids_series, np.int32)
        repeat_ids_tensor = torch.tensor(repeat_ids_array, dtype=torch.long)

        coordinates = repeats_in_sequence[["start", "end"]]
        sample, coordinates = self.transform((sample, coordinates))

        target = {
            "seq_start": [start for _ in range(coordinates.shape[0])],
            "classes": repeat_ids_tensor,
            "coordinates": coordinates,
        }
        return (sample, target)

    def seq2seq(self, index):
        sequence, start, repeats_in_sequence = self.repeat_list[index]
        end = start + self.segment_length

        repeat_ids_series = repeats_in_sequence["subtype"].map(
            self.repeat_type_mapper.label_to_index
        )
        repeat_ids_array = np.array(repeat_ids_series, np.int32)
        # repeat_ids_tensor = torch.tensor(repeat_ids_array, dtype=torch.long)

        sample = {"sequence": sequence, "start": start}

        coordinates = repeats_in_sequence[["start", "end"]]
        sample, coordinates = self.transform((sample, coordinates))
        sample = sample["sequence"]
        target = sample.clone().detach()
        for coord, c in zip(coordinates, repeat_ids_array):
            start = int(coord[0].item())
            end = int(coord[1].item())
            repeat_cls = c.item() + self.dna_sequence_mapper.num_nucleobase_letters
            target[start:end] = repeat_cls
        # <sos> target <eos>
        sos = (
            self.repeat_type_mapper.num_categories
            + self.dna_sequence_mapper.num_nucleobase_letters
        )
        eos = sos + 1
        target = torch.cat(
            (
                torch.tensor([sos], dtype=torch.long),
                target,
                torch.tensor([eos], dtype=torch.long),
            )
        )
        return sample, target

    def __getitem__(self, index):
        return self.seq2seq(index)

    def __len__(self):
        return len(self.repeat_list)

    def collate_fn(self, batch):
        sequences = [data[0]["sequence"] for data in batch]
        seq_starts = [data[0]["start"] for data in batch]
        labels = [data[1] for data in batch]
        return torch.stack(sequences), seq_starts, labels


def generate_seq2seq_dataloaders(configuration):
    dna_sequence_mapper = DnaSequenceMapper()
    dataset = RepeatSequenceDataset(
        genome_fasta_path="./data/genome_assemblies/datasets",
        annotations_path="./data/annotations",
        chromosomes=configuration.chromosomes,
        segment_length=configuration.segment_length,
        overlap=configuration.overlap,
        dna_sequence_mapper=dna_sequence_mapper,
        transform=transforms.Compose(
            [
                SampleMapEncode(dna_sequence_mapper),
                CoordinatesToTensor(),
                ZeroStartCoordinates(),
            ]
        ),
    )
    configuration.num_classes = dataset.repeat_type_mapper.num_categories
    configuration.dna_sequence_mapper = dataset.dna_sequence_mapper
    configuration.repeat_type_mapper = dataset.repeat_type_mapper
    configuration.num_nucleobase_letters = (
        configuration.dna_sequence_mapper.num_nucleobase_letters
    )

    dataset_size = len(dataset)
    if hasattr(configuration, "dataset_size"):
        dataset_size = min(dataset_size, configuration.dataset_size)
    indices = list(range(dataset_size))
    validation_size = int(configuration.validation_ratio * dataset_size)
    test_size = int(configuration.test_ratio * dataset_size)
    np.random.seed(configuration.seed)
    np.random.shuffle(indices)
    val_indices, test_indices, train_indices = (
        indices[:validation_size],
        indices[validation_size : validation_size + test_size],
        indices[validation_size + test_size : dataset_size],
    )

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=configuration.batch_size,
        sampler=train_sampler,
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=configuration.batch_size,
        sampler=valid_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=configuration.batch_size,
        sampler=test_sampler,
    )
    return train_loader, validation_loader, test_loader


def download_file(
    source_url: str, file_save_path: Union[pathlib.Path, str], chunk_size: int = 10240
):
    """
    Download a file in chunks, show progress bar while downloading.

    Args:
        source_url: URL of the file to be downloaded
        file_save_path: path to save the downloaded file
        chunk_size: chunk size in bytes, defaults to 10 kibibytes
    """
    if not isinstance(file_save_path, pathlib.Path):
        file_save_path = pathlib.Path(file_save_path)

    response = requests.get(source_url, stream=True)
    response.raise_for_status()

    file_size = int(response.headers.get("content-length", 0))

    with open(file_save_path, "wb+") as file, tqdm(
        desc=f"downloading {file_save_path.name}",
        total=file_size,
        unit="iB",
        unit_scale=True,
    ) as progress_bar:
        for stream_data in response.iter_content(chunk_size=chunk_size):
            current_size = file.write(stream_data)
            progress_bar.update(current_size)


def download_and_extract(
    target_directory: str, extracted_filename: str, source_url: str, checksum: str
):
    """Check if the file already exists, and if not, download, verify data integrity,
    and extract.

    Args:
        target_directory: path to the directory to download and extract the file
        extracted_filename: name of the extracted file
        source_url: URL of the file to be downloaded
        checksum: MD5 hash of the file
    """
    if not isinstance(target_directory, pathlib.Path):
        target_directory = pathlib.Path(target_directory)

    extracted_file_path = target_directory / extracted_filename
    if not extracted_file_path.is_file():
        compressed_file_path = target_directory / f"{extracted_filename}.gz"
        # check if the compressed file exists and verify its data integrity
        if not compressed_file_path.is_file() or not check_file_integrity(
            compressed_file_path, checksum
        ):
            download_file(source_url, compressed_file_path)
        extract_gzip(compressed_file_path, extracted_file_path)


def check_file_integrity(file_path: Union[pathlib.Path, str], checksum: str) -> bool:
    """Check the data integrity of a file, returns False if the file is corrupted
    to download again.

    Args:
        file_path: file path
        checksum: MD5 hash of the file
    """
    print("Checking file integrity \U0001FAF6\U0001F913")
    content_sum = hashlib.md5(open(file_path, "rb").read()).hexdigest()
    # verify file checksum
    res = checksum == content_sum
    if not res:
        print("Oops \U0001F928, the file has been attacked by Iron Man")
    return res


def extract_gzip(
    compressed_file_path: Union[pathlib.Path, str],
    extracted_file_path: Union[pathlib.Path, str],
):
    """Extract a gzip file.

    Args:
        compressed_file_path: gzip compressed file path
        extracted_file_path: extracted file path
    """
    print("Unzipping... \U0001F600\U0001F63C\U0001F9B2\U0001F349\U0001F34A")
    with gzip.open(compressed_file_path, "rb") as f_in:
        with open(extracted_file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def hits_to_dataframe(hits_path: Union[pathlib.Path, str]) -> pd.DataFrame:
    """Read an annotation *.hits file to a pandas DataFrame.

    Args:
        hits_path: *.hits file path
    """
    columns = hits_column_dtypes.keys()

    hits = pd.read_csv(hits_path, sep="\t", names=columns)

    # drop last row (contains the CSV header, i.e. column names)
    hits.drop(hits.tail(1).index, inplace=True)

    hits = hits.astype(hits_column_dtypes)

    return hits


class SampleMapEncode:
    def __init__(self, mapper):
        self.sequence_mapper = mapper

    def __call__(self, item):
        # [n, 2]
        sample, target_df = item
        sample["sequence"] = self.sequence_mapper.sequence_to_label_encoding(
            sample["sequence"]
        )
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


class ZeroStartCoordinates:
    """Normalize a sample's repeat annotation coordinates to a relative location
    in the sequence, defined as start and end floats between 0 and 1."""

    def __init__(self):
        pass

    def __call__(self, item):
        sample, coordinates = item
        start_coordinate = sample["start"]
        coordinates[:, :] -= start_coordinate
        return (sample, coordinates)
