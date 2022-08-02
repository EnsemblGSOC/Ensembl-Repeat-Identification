# standard library
import argparse

# project
from retrieve_annotation import retrieve_annotation
from retrieve_genome_assembly import retrieve_genome_assembly
from pytorch_lightning.utilities import AttributeDict
import yaml


def generate_datasets(args):
    with open(args.configuration) as file:
        configuration = yaml.safe_load(file)
    configuration = AttributeDict(configuration)

    retrieve_genome_assembly(args.species)
    retrieve_annotation(args.species, configuration)


def main():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--species",
        type=str,
        choices=["hg38", "mm10"],
        required=True,
        help="generate species datasets for deep learning",
    )

    parser.add_argument(
        "--configuration", type=str, help="experiment configuration file path"
    )
    args = parser.parse_args()

    generate_datasets(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
