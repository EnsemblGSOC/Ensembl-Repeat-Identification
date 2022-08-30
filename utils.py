# standard library
import gzip
import hashlib
import logging
import math
import pathlib
import shutil
import sys

from typing import Type, Union

# third party
import pandas as pd
import requests
import torch

from pyfaidx import Fasta
from pytorch_lightning.utilities import AttributeDict
from torchvision import transforms
from tqdm import tqdm

# project
from metadata import emojis


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
genomes_directory = data_directory / "genome_assemblies"
genomes_directory.mkdir(exist_ok=True)
annotations_directory = data_directory / "annotations"
annotations_directory.mkdir(exist_ok=True)

repeat_families_path = annotations_directory / "repeat_families.json"

hits_dtypes = {
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
        hits_pickle_path: Union[str, pathlib.Path],
        dna_sequence_mapper: Type[DnaSequenceMapper],
        configuration: AttributeDict,
        transform=None,
    ):
        super().__init__()

        self.dna_sequence_mapper = dna_sequence_mapper
        self.transform = transform

        self.chromosomes = configuration.chromosomes
        self.repeat_types = configuration.repeat_types
        self.segment_length = configuration.segment_length
        self.overlap = configuration.overlap

        # open genome FASTA file
        self.genome = Fasta(genome_fasta_path)

        # load genome repeats annotation and filter hits by chromosome and repeat type
        logger.info("loading genome repeats annotation...")
        hits = pd.read_pickle(hits_pickle_path)
        hits = hits.loc[hits["seq_name"].isin(self.chromosomes)]
        hits = hits.loc[hits["repeat_type"].isin(self.repeat_types)]
        self.hits = hits

        logger.info("generating repeats segments list...")
        # # TODO
        self.samples = self.generate_samples()
        # print(samples[0])
        # print(samples[-1])
        # print(len(samples))
        # exit()

        # logger.info("dataset loading complete")

        repeat_types = sorted(self.hits["repeat_type"].dropna().unique().tolist())
        self.repeat_type_mapper = CategoryMapper(repeat_types)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence, start, segment_repeats = self.samples[index]
        end = start + self.segment_length

        repeat_ids_series = segment_repeats["repeat_type"].map(
            self.repeat_type_mapper.label_to_index
        )
        repeat_ids_array = np.array(repeat_ids_series, np.int32)
        # repeat_ids_tensor = torch.tensor(repeat_ids_array, dtype=torch.long)

        sample = {"sequence": sequence, "start": start}

        coordinates = segment_repeats[["start", "end"]]
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

    def generate_samples(self):
        """
        Iterate over all segments in `chromosomes` according to `segment_length` and `overlap`,
        and save a list of the segments that contain a repeat of type in `repeat_types`.
        """
        selected_columns = ["seq_name", "ali-st", "ali-en", "repeat_type"]
        renamed_columns = ["chromosome", "start", "end", "repeat_type"]
        rename_columns_dict = dict(zip(selected_columns, renamed_columns))

        samples = []
        for chromosome in self.chromosomes:
            chromosome_length = len(self.genome[chromosome])
            num_segments = math.floor(
                (chromosome_length - self.overlap)
                / (self.segment_length - self.overlap)
            )
            for index in tqdm(
                range(num_segments), desc=f"generating {chromosome} samples"
            ):
                start = index * (self.segment_length - self.overlap)
                end = start + self.segment_length
                segment_repeats = self.get_segment_repeats(chromosome, start, end)

                if not segment_repeats.empty:
                    # breakpoint()
                    # print(segment_repeats)
                    segment_repeats = segment_repeats[selected_columns]
                    # print(segment_repeats)
                    segment_repeats = segment_repeats.rename(
                        columns=rename_columns_dict
                    )
                    # print(segment_repeats)
                    segment_repeats = segment_repeats.apply(
                        lambda x: [
                            x["chromosome"],
                            max(start, x["start"]),
                            min(end, x["end"]),
                            x["repeat_type"],
                        ],
                        axis=1,
                        result_type="broadcast",
                    )
                    # print(segment_repeats)
                    # print(self.genome[chromosome][start:end].seq)
                    samples.append(
                        (
                            self.genome[chromosome][start:end].seq.upper(),
                            start,
                            segment_repeats,
                        )
                    )
                    # print(samples[-1])
                    # if len(segment_repeats) > 2:
                    #     exit()
                    # print()
            return samples

    def get_segment_repeats(self, chromosome, start, end):
        """
        Get `chromosome` segment `start` - `end` repeat hits.

        Currently implemented only for the forward strand.
        """
        hits = self.hits

        # print(hits.head())
        # exit()

        # breakpoint()

        segment_repeats = hits.loc[
            # select chromosome
            (hits["seq_name"] == chromosome)
            # select forward strand repeats
            & (hits["ali-st"] < hits["ali-en"])
            & (
                # ----------------------------
                # ^seq_start                  ^seq_end
                #             ---------- ...
                #             ^rep_start
                ((start <= hits["ali-st"]) & (hits["ali-st"] < end))
                # ----------------------------
                # ^seq_start                  ^seq_end
                #    ...------------
                #                  ^rep_end
                | ((start < hits["ali-en"]) & (hits["ali-en"] <= end))
            )
        ]
        return segment_repeats


def generate_seq2seq_dataloaders(configuration):
    dna_sequence_mapper = DnaSequenceMapper()
    dataset = RepeatSequenceDataset(
        genome_fasta_path=genomes_directory / f"{configuration.dataset_id}.fa",
        hits_pickle_path=data_directory / f"{configuration.dataset_id}_hits.pickle",
        configuration=configuration,
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


def hits_to_dataframe(
    hits_path: Union[pathlib.Path, str], concise=False
) -> pd.DataFrame:
    """Read an annotation *.hits file to a pandas DataFrame.

    Args:
        hits_path: *.hits file path
    """
    columns = hits_dtypes.keys()

    use_columns = [
        "seq_name",
        "family_acc",
        "family_name",
        "strand",
        "ali-st",
        "ali-en",
    ]

    if concise:
        hits = pd.read_csv(
            hits_path, sep="\t", names=columns, usecols=use_columns, dtype=hits_dtypes
        )
    else:
        hits = pd.read_csv(hits_path, sep="\t", names=columns, dtype=hits_dtypes)

    return hits


class CategoryMapper:
    """
    Categorical data mapping class, with methods to translate from the category
    text labels to one-hot encoding and vice versa.
    """

    def __init__(self, categories):
        self.categories = sorted(categories)
        self.num_categories = len(self.categories)
        self.emojis = emojis[: self.num_categories + 2]
        self.label_to_index_dict = {
            label: index for index, label in enumerate(categories)
        }
        self.index_to_label_dict = {
            index: label for index, label in enumerate(categories)
        }
        self.index_to_emoji_dict = {
            index: emoji for index, emoji in enumerate(self.emojis)
        }

        self.label_to_emoji_dict = {
            label: self.index_to_emoji_dict[index]
            for index, label in enumerate(categories + ["sos", "eos"])
        }

    def label_to_index(self, label):
        """
        Get the class index of label.
        """
        return self.label_to_index_dict[label]

    def index_to_label(self, index):
        """
        Get the label string from its class index.
        """
        return self.index_to_label_dict[index]

    def label_to_emoji(self, index):
        return self.index_to_emoji_dict[index]

    def label_to_one_hot(self, label):
        """
        Get the one-hot representation of label.
        """
        one_hot_label = F.one_hot(
            torch.tensor(self.label_to_index_dict[label]),
            num_classes=self.num_categories,
        )
        one_hot_label = one_hot_label.type(torch.float32)
        return one_hot_label

    def one_hot_to_label(self, one_hot_label):
        """
        Get the label string from its one-hot representation.
        """
        index = torch.argmax(one_hot_label)
        label = self.index_to_label_dict[index]
        return label

    def print_label_and_emoji(self, logger):
        logger.info(self.label_to_emoji_dict)


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
