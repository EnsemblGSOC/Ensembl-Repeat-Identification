# standard library
import gzip
import hashlib
import logging
import pathlib
import shutil
import sys

from typing import Union

# third party
import pandas as pd
import requests

from tqdm import tqdm


data_directory = pathlib.Path("data")
data_directory.mkdir(exist_ok=True)


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
        desc=file_save_path.name,
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
        un_gz(compressed_file_path, extracted_file_path)


def check_file_integrity(file_path: str, checksum: str) -> bool:
    """Check the data integrity of a file, returns False if the file is corrupted
    to download again.

    Args:
        file_path: file path
        checksum: MD5 hash of the file
    """
    print("Checking file integrity \U0001FAF6\U0001F913")
    content_sum = hashlib.md5(open(file_path, "rb").read()).hexdigest()
    # check the fasta.gz size
    res = checksum == content_sum
    if not res:
        print("Oops \U0001F928, the file have been attack by Iron Man")
    return res


def un_gz(zipped: str, unzipped: str):
    """unzip the gz files.

    Args:
        unzipped - the unzipped name of file
            e.g. hg38.fa
        zipped - the zipped name of file
            e.g. hg38.gz.fa
    """
    print("Unziping... \U0001F600\U0001F63C\U0001F9B2\U0001F349\U0001F34A")
    with gzip.open(zipped, "rb") as f_in:
        with open(unzipped, "wb") as f_out:
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
