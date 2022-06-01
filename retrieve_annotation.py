# standard library
import csv
import json
import pathlib

from typing import NamedTuple, Union

# third party
import requests

# project
from config import chr_length, species_integrity, url_label_information
from utils import data_directory, download_and_extract


class AnnotationInfo(NamedTuple):
    chromosome: str
    subtype: str
    start: int
    end: int


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


def retrieve_annotation(assembly: str):
    """Download the assembly annotation and convert to appropriate format.

    Args:
        assembly: genome assembly name used by Dfam (e.g. hg38)
    """
    # set and create the annotations directory
    annotations_directory = data_directory / "annotations"
    annotations_directory.mkdir(exist_ok=True)

    # download Dfam repeat families
    repeat_families_path = annotations_directory / "repeat_families.json"
    if not repeat_families_path.is_file():
        download_repeat_families(repeat_families_path)

    # load repeat families
    with open(repeat_families_path) as json_file:
        repeat_families = json.load(json_file)

    # download and extract the original annotations file
    checksum = species_integrity[f"{assembly}.hits"]
    download_and_extract(
        annotations_directory,
        f"{assembly}.hits",
        url_label_information[assembly],
        checksum,
    )

    wanted = extract_lines(f"{annotations_directory}/{assembly}.hits", repeat_families)
    for chromosome, length in chr_length.items():
        data = list(filter(lambda x: x.chromosome.split(":")[0] == chromosome, wanted))
        save_annotations(annotations_directory, assembly, chromosome, data)


def extract_lines(assembly_fasta_path: str, repeat_families: dict):
    """match the information of web with the hits files

    Args:
        assembly_fasta_path: genome assembly FASTA path
            e.g. data/genome_assemblies/hg38.fa
        repeat_families: repeat families dictionary
    """
    print("Generating label datasets \U0001F43C\U0001F43E\U0001F43E")
    wanted = []
    num_skipped_repeats = 0
    with open(assembly_fasta_path, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        for data in reader:
            accession = data[1]
            if not data[2].startswith("LTR"):
                continue
            subtype = repeat_families[accession]["classification"]
            ali_start, ali_end = int(data[9]), int(data[10])
            seq_start, seq_end = get_corresponding_sequence_position(ali_start, ali_end)
            # repeated sequence will be skipped:
            # ｜----------｜-----------｜ chromosome segments position
            #       ｜--------｜ repeated sequence position
            if not (ali_start >= seq_start and ali_end <= seq_end):
                num_skipped_repeats += 1
                continue
            name = f"{data[0]}:{seq_start+1}-{seq_end}"
            wanted.append(
                AnnotationInfo(
                    chromosome=name,
                    subtype=subtype,
                    start=data[9],
                    end=data[10],
                )
            )

    # count all boundary repeat sequence numbers
    print(f"{num_skipped_repeats} repeated sequences have been removed")

    return wanted


def get_corresponding_sequence_position(ali_start: int, ali_end: int):
    """add chromosome information matching the FASTA segment position

    Args:
        ali_start: start with chromosome position
        ali_end: end with chromosome position
    """
    interval = 100000
    sequence_start = ali_start // interval * interval
    sequence_end = sequence_start + interval
    return (sequence_start, sequence_end)


def save_annotations(
    assemblies_directory: str, assembly: str, chromosome: str, annotations: list
):
    """make files to save the new datasets

    Args:
        assemblies_directory: assemblies directory path
            e.g. data/genome_assemblies
        assembly: genome assembly name used by Dfam (e.g. hg38)
        chromosome: chromosome name (e.g. chr1)
        annotations: the target region, with its own alignment star, end and type.
    """
    annotations_csv = f"{assemblies_directory}/{assembly}_{chromosome}.csv"
    with open(annotations_csv, "a+", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter="\t", lineterminator="\n")
        csv_writer.writerows(
            [
                (data.chromosome, data.start, data.end, data.subtype)
                for data in annotations
            ]
        )
