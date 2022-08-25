# standard library
import argparse

# third party
import yaml

from pytorch_lightning.utilities import AttributeDict

# project
from metadata import genomes
from retrieve_annotation import retrieve_annotation
from utils import genome_assemblies_directory, download_and_extract


def retrieve_genome_assembly(assembly: str):
    """Download (reference) genome assembly and convert to appropriate format.

    Args:
        assembly: genome assembly name used by Dfam (e.g. hg38)
    """
    download_and_extract(
        genome_assemblies_directory,
        f"{assembly}.fa",
        genomes[assembly]["genome_url"],
        genomes[assembly]["genome_checksum"],
    )


def main():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--species",
        type=str,
        choices=["hg38", "mm10"],
        required=True,
        help="generate species datasets for deep learning",
    )

    args = parser.parse_args()

    retrieve_genome_assembly(args.species)
    retrieve_annotation(args.species)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
