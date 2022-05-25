
# standard library
import csv

from chr_info import chr_length


def chr_to_bed(species):
    for chr, length in chr_length.items():
        pos = []
        for i in range(1, length, 100000):
            end = i + 100000
            if end > length:
                end = length
            pos.append((chr, i, end))
        save_bed_file(species, chr, pos)


def save_bed_file(species: str, chr: str, pos: list):
    chr_bed_file = f"bed_files/{species}_{chr}.bed"
    with open(chr_bed_file, "a+", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter="\t", lineterminator="\n")
        csv_writer.writerows(pos)
