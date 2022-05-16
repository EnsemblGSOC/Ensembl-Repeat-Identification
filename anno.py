"""
Author: yangtcai yangtcai@gmail.com
Date: 2022-05-12 22:31:30
LastEditors: yangtcai yangtcai@gmail.com
LastEditTime: 2022-05-16 17:53:03
FilePath: /undefined/Users/caiyz/Desktop/anno
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""


import argparse
import csv
import functools
import json

import requests

from tqdm import tqdm

from chr_info import chr_length


# this cache shouldn't exceed available memory, since the number of families is less than a million
@functools.cache
def get_subtype(accession_id: str):
    url = "https://dfam.org/api/families/" + accession_id
    response = requests.get(url)
    return response.json()["classification"]


def get_annotation(sp, chr, op, ed):
    url = "https://dfam.org/api/annotations"
    params = {
        "assembly": sp,
        "chrom": chr,
        "start": op,
        "end": ed,
    }

    response = requests.get(url, params=params)
    results = response.json()
    annotations = []
    for hit in results["hits"]:
        if hit["type"] == "LTR":
            annotations.append(
                [chr, hit["ali_start"], hit["ali_end"], get_subtype(hit["accession"])]
            )

    save_to_csv(annotations)


def save_to_csv(annotations):
    with open("test.csv", "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(annotations)


# get_annotation('hg38', 'chr3', 147733000, 147766820)


def get_all(args):
    for chr_name, chr_len in chr_length.items():

        for i in tqdm(range(0, chr_len, 100000)):
            get_annotation(args.species, chr_name, i, i + 100000)


if __name__ == "__main__":
    # execute only if run as a script

    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--species",
        type=str,
        choices=["hg38", "mm10"],
        required=True,
        help="which species you interested in",
    )

    args = parser.parse_args()
    get_all(args)
