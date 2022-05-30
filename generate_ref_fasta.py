# standard library
import csv

# third party
import pybedtools
from pathlib import Path

# project
from config import chr_length, url_species, species_integrity
from requests_progress_bar import ProgressBar
from utils import download_and_unzip, mkdir


def download_fasta_ref(species: str):
    """all the fasta information generated from there to start.

    Keyword arguments:
    species - the name of reference genome.
    e.g. hg38
    """
    folder = "./ref_datasets/"
    mkdir(folder)
    checksum = species_integrity[species]
    download_and_unzip(species, folder, f"{species}.fa", url_species[species], checksum)
    chr_to_bed(species, f"{folder}/{species}.fa")


def chr_to_bed(species: str, fasta_filename):
    """generate the bed of chromosome to generate the fasta

    Keyword arguments:
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
            pos.append("\t".join([str(chr), str(i), str(end)]))
        use_bedtools(species, chr, "\n".join(pos), fasta_filename)


def use_bedtools(species: str, chr: str, chr_fasta: str, fasta_filename: str):
    """use pybedtools to generate the fasta sequence

    Keyword arguments:
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

    Keyword arguments:
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

    Keyword arguments:
    species -  the name of reference genome.
    e.g. hg38
    chr - the chromosome of the chosen species.
    e.g. chr1
    chr_withlines -  the information from transfer_fasta().
    """
    folder = "./ref_datasets/datasets/"
    mkdir(folder)
    transfer_to_datasets = f"{folder}/{species}_{chr}_ref.csv"
    with open(transfer_to_datasets, "w+", newline="") as ttd:
        csv_writer = csv.writer(ttd, delimiter="\t", lineterminator="\n")
        csv_writer.writerow(chr_withlines)


if __name__ == "__main__":
    download_fasta_ref("hg38")
