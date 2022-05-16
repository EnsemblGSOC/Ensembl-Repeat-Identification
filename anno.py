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
def get_classification(accession_id: str):
    url = f"https://dfam.org/api/families/{accession_id}"

    response = requests.get(url)
    response.raise_for_status()

    classification = response.json()["classification"]

    return classification


def get_range_annotations(assembly: str, chromosome: str, start: int, end: int):
    """
    https://www.dfam.org/releases/Dfam_3.6/apidocs/#operation--annotations-get

    e.g.
    get_range_annotations("hg38", "chr3", 147733000, 147766820)
    """
    relevant_repeat_classes = ["LTR"]

    url = "https://dfam.org/api/annotations"
    params = {
        "assembly": assembly,
        "chrom": chromosome,
        "start": start,
        "end": end,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    hits = response.json()["hits"]

    annotations = []
    for hit in hits:
        if hit["type"] in relevant_repeat_classes:
            annotations.append(
                [
                    chromosome,
                    hit["ali_start"],
                    hit["ali_end"],
                    get_classification(hit["accession"]),
                ]
            )

    return annotations


def download_species_annotations(species: str):
    for chromosome, length in chr_length.items():
        for i in tqdm(range(0, length, 100000)):
            annotations = get_range_annotations(species, chromosome, i, i + 100000)
            save_annotations(species, chromosome, annotations)


def save_annotations(species: str, chromosome: str, annotations: list):
    annotations_csv = f"{species}_{chromosome}.csv"
    with open(annotations_csv, "a+", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter="\t", lineterminator="\n")
        csv_writer.writerows(annotations)


def main():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--species",
        type=str,
        choices=["hg38", "mm10"],
        required=True,
        help="species for which to download annotations",
    )

    args = parser.parse_args()

    download_species_annotations(args.species)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
