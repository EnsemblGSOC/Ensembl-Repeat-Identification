# Ensembl-Repeat-Identification


## Usage

Set up a development environment with [pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://github.com/python-poetry/poetry):
```shell
pyenv install 3.9.13

pyenv virtualenv 3.9.13 repeat_identification

poetry install
```

### Generate the target repeat sequence datasets

```shell
python generate_dataset.py --genome_assembly <UCSC genome assembly ID>

# e.g.
python generate_dataset.py --genome_assembly hg38
```
