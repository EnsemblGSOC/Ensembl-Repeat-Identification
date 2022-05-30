# standard library
import argparse

# project
from generate_label import download_annotation
from generate_ref_fasta import download_fasta_ref


def generate_datasets(species: str):
    download_annotation(species)
    download_fasta_ref(species)


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
