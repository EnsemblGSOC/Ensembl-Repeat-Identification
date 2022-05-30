

# Ensembl-Repeat-Identification

## Usage

Set up a development environment with [pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://github.com/python-poetry/poetry):
```shell
pyenv install 3.9.12

pyenv virtualenv 3.9.12 repeat_identification

poetry install

```

### Generate the target repeat sequence datasets.

```shell
python3 generate_datasets --species

e.g. python3 generate_datasets --hg38
```