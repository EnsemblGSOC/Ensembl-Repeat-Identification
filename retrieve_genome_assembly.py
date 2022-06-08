# standard library
import csv

# third party
from Bio import SeqIO

# project
from config import chr_length, species_integrity, url_species
from utils import data_directory, download_and_extract


def retrieve_genome_assembly(assembly: str):
    """Download (reference) genome assembly and convert to appropriate format.

    Args:
        assembly: genome assembly name used by Dfam (e.g. hg38)
    """
    assemblies_directory = data_directory / "genome_assemblies"
    assemblies_directory.mkdir(exist_ok=True)

    checksum = species_integrity[assembly]
    download_and_extract(
        assemblies_directory, f"{assembly}.fa", url_species[assembly], checksum
    )
    separate_FASTA(assemblies_directory / f"{assembly}.fa")


def separate_FASTA(assembly):
    """Split (reference) genome assembly into 'chromosome.fa' files.

    Args:
        assembly: genome assembly name used by Dfam (e.g. hg38)
    """
    directory = data_directory / "genome_assemblies" / "datasets"
    directory.mkdir(exist_ok=True)
    print("Generating reference datasets \U0001F95D\U0001F353\U0001F364\U0001F95F")
    records = SeqIO.parse(assembly, "fasta")
    # parse can use for multi-name task.
    for record in records:
        for chromosome, _ in chr_length.items():
            if record.name == chromosome:
                SeqIO.write(record, directory / f"{chromosome}.fa", "fasta")


if __name__ == "__main__":
    retrieve_genome_assembly("hg38")
