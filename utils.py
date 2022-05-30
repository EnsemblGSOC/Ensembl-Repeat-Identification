"""
Author: yangtcai yangtcai@gmail.com
Date: 2022-05-30 13:05:15
LastEditors: yangtcai yangtcai@gmail.com
LastEditTime: 2022-05-30 22:25:38
FilePath: /Ensembl-Repeat-Identification/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

# standard library
import gzip
import hashlib
import pathlib
import shutil

from typing import Union

# third party
import requests

from tqdm import tqdm


def download_file(
    source_url: str, save_path: Union[pathlib.Path, str], chunk_size: int = 10240
):
    """
    Download a file in chunks, show progress bar while downloading.

    Args:
        source_url: URL of the file to be downloaded
        save_path: path to save the downloaded file
        chunk_size: chunk size in bytes, defaults to 10 kibibytes
    """
    if not isinstance(save_path, pathlib.Path):
        save_path = pathlib.Path(save_path)

    response = requests.get(source_url, stream=True)
    response.raise_for_status()

    file_size = int(response.headers.get("content-length", 0))

    with open(save_path, "wb+") as file, tqdm(
        desc=save_path.stem,
        total=file_size,
        unit="iB",
        unit_scale=True,
    ) as progress_bar:
        for stream_data in response.iter_content(chunk_size=chunk_size):
            current_size = file.write(stream_data)
            progress_bar.update(current_size)


def download_and_unzip(
    species: str, folder: str, filename: str, url: str, checksum: str
):
    """check if fasta exist, if not download fasta, and make a new file.

    Args:
        species - the name of reference genome
            e.g. hg38
        filename - the name of unziped file, like hg38.fa
        checksum - the checksum of the file, to verify data integrity
    """
    unziped_file = pathlib.Path(f"{folder}/{filename}")
    ziped_file = pathlib.Path(f"{folder}/{filename}.gz")
    if not unziped_file.exists():
        if not ziped_file.exists() or not check_integrity(
            ziped_file, checksum
        ):  # check if exist the fasta.gz and the integrity of the fasta.gz
            download_file(url, ziped_file)
        un_gz(ziped_file, unziped_file)


def check_integrity(ziped_file: str, checksum: str) -> bool:
    """check the integrity of already downloand reference fasta, if it not integrate, download again.

    Args:
        species - the name of reference genome
            e.g. hg38
        species_gz_ref - the name of fasta.gz
    """
    print("Checking file integrity\U0001FAF6\U0001F913")
    content_sum = hashlib.md5(open(ziped_file, "rb").read()).hexdigest()
    # check the fasta.gz size
    res = checksum == content_sum
    if not res:
        print("Oops\U0001F928, the file have been attack by Iron Man")
    return res


def un_gz(zipped: str, unzipped: str):
    """unzip the gz files.

    Args:
        unzipped - the unzipped name of file
            e.g. hg38.fa
        zipped - the zipped name of file
            e.g. hg38.gz.fa
    """
    print("Unziping……\U0001F600\U0001F63C\U0001F9B2\U0001F349\U0001F34A")
    with gzip.open(zipped, "rb") as f_in:
        with open(unzipped, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
