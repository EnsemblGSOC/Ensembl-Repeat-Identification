# standard library
import csv
import pathlib

# third party
import pybedtools

# project
from config import chr_length, species_integrity, url_species
from utils import download_and_unzip


def download_fasta_ref(species: str):
    """all the fasta information generated from there to start.

    Args:
        species - the name of reference genome.
            e.g. hg38
    """
    directory = pathlib.Path("ref_datasets")
    directory.mkdir(exist_ok=True)
    checksum = species_integrity[species]
    download_and_unzip(
        species, directory, f"{species}.fa", url_species[species], checksum
    )
    chr_to_bed(species, f"{directory}/{species}.fa")


def chr_to_bed(species: str, fasta_filename):
    """generate the bed of chromosome to generate the fasta

    Args:
        fasta_filename - whole path with species information.
            e.g. ./ref_datasets/hg38.fa
        species - the name of reference genome,
            e.g. hg38
    """
    print("Genertating reference datasets\U0001F95D\U0001F353\U0001F364\U0001F95F")
    for chr, length in chr_length.items():
        pos = []
        for i in range(1, length, 100000):
            end = i + 100000
            if end > length:
                end = length
            pos.append("\t".join([chr, str(i), str(end)]))
        use_bedtools(species, chr, "\n".join(pos), fasta_filename)


def use_bedtools(species: str, chr: str, chr_fasta: str, fasta_filename: str):
    """use pybedtools to generate the fasta sequence

    Args:
        species - the name of reference genome,
            e.g. hg38
        chr_fasta - the real sequence position bed
        chr - chromosome name
            e.g. chr1
        fasta_filename - whole path with species information.
            e.g. ./ref_datasets/hg38.fa
    """

    bedtools_information = pybedtools.BedTool(chr_fasta, from_string=True)
    bedtools_information = bedtools_information.sequence(fi=fasta_filename)
    seq_bed = open(bedtools_information.seqfn).read()
    transfer_fasta(seq_bed, species, chr)


def transfer_fasta(seq_bed: str, species: str, chr: str):
    """format the fasta file for next step

    Args:
        seq_bed - the generation of use_bedtools()
        chr - chromosome name
            e.g. chr1
        species - the name of reference genome,
            e.g. hg38
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

    fasta_lines(chr_withlines, species, chr)


def fasta_lines(chr_withlines: list, species: str, chr: str):
    """make files to save the new datasets

    Args:
        species -  the name of reference genome.
            e.g. hg38
        chr - the chromosome of the chosen species.
            e.g. chr1
        chr_withlines -  the information from transfer_fasta().
    """
    directory = pathlib.Path("ref_datasets/datasets")
    directory.mkdir(exist_ok=True)
    transfer_to_datasets = f"{directory}/{species}_{chr}_ref.csv"
    with open(transfer_to_datasets, "w+", newline="") as ttd:
        csv_writer = csv.writer(ttd, delimiter="\t", lineterminator="\n")
        csv_writer.writerow(chr_withlines)
