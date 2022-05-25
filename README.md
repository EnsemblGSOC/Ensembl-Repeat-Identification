
# Ensembl-Repeat-Identification

## Usage

Set up a development environment with [pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://github.com/python-poetry/poetry):
```shell
pyenv install 3.9.12

pyenv virtualenv 3.9.12 repeat_identification

poetry install

```

### Download annotations with the following command

```shell
python anno.py --species hg38
```


### Extract fasta sequence information

1. Install bedtools and download hg38.fa

2. Generate bed files
```shell
python anno.py --species hg38 --sequence
```
3. Generate chromosome fasta

```shell
find . -name "*.bed" | xargs -I {} bedtools getfasta -fi hg38.fa -bed {} -fo {}.fa
```
4. Transfer fasta format into text file

to do 