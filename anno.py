

# standard library
import argparse
import csv
import json
import pathlib

from typing import Union

# third party
import requests

from tqdm import tqdm

# project
from chr_info import chr_length


def get_range_annotations(
    assembly: str, chromosome: str, start: int, end: int, families: dict
):
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
                    families[hit["accession"]]["classification"],
                ]
            )

    return annotations


def download_species_annotations(species: str, families: dict):
    for chromosome, length in chr_length.items():
        for i in tqdm(range(0, length, 100000)):
            annotations = get_range_annotations(
                species, chromosome, i, i + 100000, families
            )
            save_annotations(species, chromosome, annotations)


def save_annotations(species: str, chromosome: str, annotations: list):
    annotations_csv = f"{species}_{chromosome}.csv"
    with open(annotations_csv, "a+", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter="\t", lineterminator="\n")
        csv_writer.writerows(annotations)


def download_families(families_path: Union[str, pathlib.Path]):
    """
    https://www.dfam.org/releases/Dfam_3.6/apidocs/#operation--families-get
    """
    base_url = "https://dfam.org/api/families"
    limit = 1000

    start = 1
    end = float("inf")

    print("downloading Dfam families...")

    families = {}
    while start < end:
        url = f"{base_url}?start={start}&limit={limit}"

        response = requests.get(url)
        response.raise_for_status()

        response_json = response.json()

        families_batch = response_json["results"]
        for family in families_batch:
            accession = family["accession"]
            assert accession not in families, f"duplicate accession ID {accession}"
            families[accession] = family

        print(f"{len(families)}")

        start += limit
        end = response_json["total_count"]

    print(f"{len(families)} total families downloaded")

    with open(families_path, "w") as json_file:
        json.dump(families, json_file)


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

    families_filename = "families.json"
    families_path = pathlib.Path(families_filename)
    if not families_path.is_file():
        download_families(families_path)

    with open(families_path) as json_file:
        families = json.load(json_file)

    download_species_annotations(args.species, families)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
