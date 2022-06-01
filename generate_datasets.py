# standard library
import argparse

# project
from retrieve_annotation import retrieve_annotation
from retrieve_genome_assembly import retrieve_genome_assembly


def generate_datasets(species: str):
    retrieve_genome_assembly(species)
    retrieve_annotation(species)


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

    generate_datasets(args.species)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
