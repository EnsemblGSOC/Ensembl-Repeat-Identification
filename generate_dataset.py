# standard library
import argparse
import json
import os
import pathlib
import subprocess

from typing import Union

# third party
import requests

# project
from metadata import genomes
from utils import (
    annotations_directory,
    data_directory,
    download_and_extract,
    genomes_directory,
    hits_to_dataframe,
    logger,
    repeat_families_path,
)


def retrieve_genome_assembly(genome_assembly: str):
    """Download the genome assembly.

    Args:
        genome_assembly: genome assembly name used by Dfam (e.g. hg38)
    """
    download_and_extract(
        genomes_directory,
        f"{genome_assembly}.fa",
        genomes[genome_assembly]["genome_url"],
        genomes[genome_assembly]["genome_checksum"],
    )


def retrieve_annotation(genome_assembly: str):
    """Download the genome assembly repeats annotation and convert to appropriate format.

    Args:
        genome_assembly: genome assembly name used by Dfam (e.g. hg38)
    """
    # download Dfam repeat families
    if not repeat_families_path.is_file():
        download_repeat_families(repeat_families_path)

    # load repeat families
    with open(repeat_families_path) as json_file:
        repeat_families = json.load(json_file)

    # download and extract the original annotation file
    download_and_extract(
        annotations_directory,
        f"{genome_assembly}.hits",
        genomes[genome_assembly]["annotation_url"],
        genomes[genome_assembly]["annotation_checksum"],
    )

    annotation_path = annotations_directory / f"{genome_assembly}.hits"
    last_line = tail(annotation_path, n=1)[-1]
    if "seq_name" in last_line:
        # Delete the last line of the annotation hits file. That line contains the column names
        # and overcomplicates loading the file to a DataFrame.

        logger.info("deleting repeats annotation file last line with column names...")
        delete_last_line(annotation_path)


def download_repeat_families(repeat_families_path: Union[str, pathlib.Path]):
    """Download all Dfam repeat families and save to a JSON file.

    https://www.dfam.org/releases/Dfam_3.6/apidocs/#operation--families-get

    Args:
        repeat_families_path: repeat families JSON file path
    """
    base_url = "https://dfam.org/api/families"
    limit = 1000

    start = 1
    end = float("inf")

    print("downloading Dfam repeat families...")

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

    print(f"{len(families)} total repeat families downloaded")

    with open(repeat_families_path, "w") as json_file:
        json.dump(families, json_file)


def generate_hits_dataframe_pickle(genome_assembly):
    """
    Read the repeats annotation / hits file to a DataFrame and add repeat type and
    subtype names.

    Args:
        genome_assembly: genome assembly name used by Dfam (e.g. hg38)
    """
    genome_fasta_path = genomes_directory / f"{genome_assembly}.fa"
    annotation_path = annotations_directory / f"{genome_assembly}.hits"

    logger.info("loading repeats annotation...")
    hits = hits_to_dataframe(annotation_path, concise=True)

    logger.info("loading repeat families dictionary...")
    with open(repeat_families_path) as json_file:
        families_dict = json.load(json_file)

    logger.info(
        "adding repeat type and subtype names to the repeats annotation DataFrame..."
    )
    hits["repeat_type"] = (
        hits[~hits["family_acc"].map(families_dict).isna()]["family_acc"]
        .map(families_dict)
        .apply(lambda x: x.get("repeat_type_name"))
    )
    hits["repeat_subtype"] = (
        hits[~hits["family_acc"].map(families_dict).isna()]["family_acc"]
        .map(families_dict)
        .apply(lambda x: x.get("repeat_subtype_name"))
    )

    # save hits DataFrame to a pickle file
    hits_pickle_path = data_directory / f"{genome_assembly}_hits.pickle"
    hits.to_pickle(hits_pickle_path)
    logger.info(f"repeats annotation saved at {hits_pickle_path}")


def tail(file_path, n=5):
    """
    Read the last n lines of a file and return them in a list.

    Args:
        file_path: file path
    """
    tail_command = ["tail", "-n", f"{n}", file_path]
    result = subprocess.run(tail_command, capture_output=True, text=True)
    if result.stdout[-1] == "\n":
        output = result.stdout[:-1]
    else:
        output = result.stdout
    lines = output.split("\n")
    return lines


def delete_last_line(file_path: Union[str, pathlib.Path]):
    """
    Delete the last line of a text file.

    Args:
        file_path: text file path
    """
    with open(file_path, "r+", encoding="utf-8") as file:
        # reposition file offset to the end of the file
        file.seek(0, os.SEEK_END)

        # set position to the second last character of the file
        position = file.tell() - 1

        # search backwards for the newline character
        while position > 0 and file.read(1) != "\n":
            position -= 1
            file.seek(position, os.SEEK_SET)

        # delete the subsequent characters if a newline character was found after
        # the first character
        if position > 0:
            file.seek(position + 1, os.SEEK_SET)
            file.truncate()


def main():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--genome_assembly",
        type=str,
        choices=["hg38", "mm10"],
        required=True,
        help="generate repeats dataset for deep learning",
    )

    args = parser.parse_args()

    retrieve_genome_assembly(args.genome_assembly)
    retrieve_annotation(args.genome_assembly)

    generate_hits_dataframe_pickle(args.genome_assembly)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
