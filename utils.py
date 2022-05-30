"""
Author: yangtcai yangtcai@gmail.com
Date: 2022-05-30 13:05:15
LastEditors: yangtcai yangtcai@gmail.com
LastEditTime: 2022-05-30 22:25:38
FilePath: /Ensembl-Repeat-Identification/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
# standard library
import csv
import os
import gzip
import shutil
import hashlib

# third party
import requests
from pathlib import Path

# project
from requests_progress_bar import ProgressBar
from config import chr_length, url_species, species_integrity


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.mkdir(path)


def download_file(url: str, file_name: str):
    """only for first download users.

    Keyword arguments:
    url - the url of the genome or hits
    file_name - the whole path of files
    e.g. ./ref_datasets/hg38.gz
    """

    with requests.get(url, stream=True) as r:
        chunk_size = 1024  # max of single time request
        content_size = int(r.headers["content-length"])  # content sizel
        progress = ProgressBar(
            file_name,
            total=content_size,
            unit="KB",
            chunk_size=chunk_size,
            run_status="downloading",
            fin_status="finish",
        )  # add progress bar :DDD
        with open(file_name, "wb") as f:
            for data in r.iter_content(chunk_size=chunk_size):
                f.write(data)
                progress.refresh(count=len(data))


def download_and_unzip(
    species: str, folder: str, filename: str, url: str, checksum: str
):
    """check if fasta exist, if not download fasta, and make a new file.

    Keyword arguments:
    species - the name of reference genome
    e.g. hg38
    filename - the name of unziped file, like hg38.fa
    checksum - the check file, to check size of gz file
    """
    unziped_file = Path(f"{folder}/{filename}")
    ziped_file = Path(f"{folder}/{filename}.gz")
    if not unziped_file.exists():
        if not ziped_file.exists() or not check_integrity(
            ziped_file, checksum
        ):  # check if exist the fasta.gz and the integrity of the fasta.gz
            download_file(url, ziped_file)
        un_gz(ziped_file, unziped_file)


def check_integrity(ziped_file: str, checksum: str) -> bool:
    """check the integrity of already downloand reference fasta, if it not integrate, download again.

    Keyword arguments:
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

    Keyword arguments:
    unzipped - the unzipped name of file
    e.g. hg38.fa
    zipped - the zipped name of file
    e.g. hg38.gz.fa
    """
    print("Unziping……\U0001F600\U0001F63C\U0001F9B2\U0001F349\U0001F34A")
    with gzip.open(zipped, "rb") as f_in:
        with open(unzipped, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
