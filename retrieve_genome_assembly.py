# standard library
import csv

# third party
import pybedtools

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
    chr_to_bed(assembly, f"{assemblies_directory}/{assembly}.fa")


def chr_to_bed(assembly: str, fasta_path):
    """generate the bed of chromosome to generate the fasta

    Args:
        assembly: genome assembly name used by Dfam (e.g. hg38)
        fasta_path: genome assembly FASTA path
            e.g. data/genome_assemblies/hg38.fa
    """
    print("Generating reference datasets \U0001F95D\U0001F353\U0001F364\U0001F95F")
    for chromosome, length in chr_length.items():
        coordinates = []
        for start in range(1, length, 100000):
            end = min(start + 100000 - 1, length)
            coordinates.append("\t".join([chromosome, str(start), str(end)]))
        generate_subsequences_fasta(
            assembly, chromosome, "\n".join(coordinates), fasta_path
        )


def generate_subsequences_fasta(
    assembly: str, chromosome: str, chromosome_fasta: str, fasta_path: str
):
    """Use pybedtools to generate the fasta sequence.

    Args:
        assembly: genome assembly name used by Dfam (e.g. hg38)
        chromosome: chromosome name
            e.g. chr1
        chromosome_fasta - the real sequence position bed
        fasta_path: genome assembly FASTA path
    """
    bedtools_information = pybedtools.BedTool(chromosome_fasta, from_string=True)
    bedtools_information = bedtools_information.sequence(fi=fasta_path)
    seq_bed = open(bedtools_information.seqfn).read()
    transfer_fasta(seq_bed, assembly, chromosome)


def transfer_fasta(seq_bed: str, assembly: str, chromosome: str):
    """format the fasta file for next step

    Args:
        seq_bed - the generation of generate_subsequences_fasta()
        assembly: genome assembly name used by Dfam (e.g. hg38)
        chromosome - chromosome name
            e.g. chr1
    """
    chr_withlines = []
    l = []
    seq_dataset = seq_bed.split("\n")
    for GCTA_line in seq_dataset:
        if GCTA_line.startswith(">"):
            if len(l) == 0:
                l.append(GCTA_line.strip(">"))
            else:
                chr_withlines.append(",".join(l))
                l = []
        else:
            l.append(GCTA_line.strip("\n"))

    fasta_lines(chr_withlines, assembly, chromosome)


def fasta_lines(chr_withlines: list, assembly: str, chromosome: str):
    """make files to save the new datasets

    Args:
        assembly: genome assembly name used by Dfam (e.g. hg38)
        chromosome - the chromosome of the chosen assembly.
            e.g. chr1
        chr_withlines -  the information from transfer_fasta().
    """
    directory = data_directory / "genome_assemblies" / "datasets"
    directory.mkdir(exist_ok=True)

    transfer_to_datasets = f"{directory}/{assembly}_{chromosome}_ref.csv"
    with open(transfer_to_datasets, "w+", newline="") as ttd:
        csv_writer = csv.writer(ttd, delimiter="\t", lineterminator="\n")
        csv_writer.writerow(chr_withlines)
