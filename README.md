# Google Summer of Code-2022 : Ensembl-Repeat-Identification
#### A Deep Learning repository for predicting the `location` and `type` of repeat sequence in genome.

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)

[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://github.com/idealo/image-super-resolution/blob/master/LICENSE)
### Mentors : [Leanne Haggerty](https://www.linkedin.cn/incareer/in/leannehaggerty), [William Stark](https://www.linkedin.cn/incareer/in/williamstarkbio), [Jose Perez-Silva](https://www.linkedin.cn/incareer/in/jos%C3%A9-m%C2%AA-g-p%C3%A9rez-silva-b3959386), [Francesca Tricomi](https://www.linkedin.cn/incareer/in/francesca-tricomi-108916168) <br/><br/>

## Brief Description
A number of tools exist for identifying repeat features, but it remains a problem that the DNA sequence of some genes can be identified as being a repeat sequence. If such sequences are used to mask the genome, genes may be missed in the downstream annotation. Assuming that gene sequences have various signatures relating to their function and that repeats have different signatures including the repetitive nature of the signal itself, we want to train a classifier to separate the repeat sequences from the gene sequences. We are inspired by DETR, an object detection model, this proposal will use transformer structure to complete the identify repeat sequence task, our model will unify segmentation and classification into one like the object detection model.
## Network architecture
![](fig/DETRmodel.png)
We need input the sequence as input, and the output will be the location and type of repeat.
## Requirements:
1. A machine with atleast **8GB of RAM** (although **16-32GB** is recommended. A single GPU machine would suffice. The model can be trained on CPU as well but will be a lot faster if trained on a GPU.<br/><br/>
2. A stable Internet Connection.<br/><br/>
3. Set up a development environment with [pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://github.com/python-poetry/poetry):
```shell
pyenv install 3.9.12

pyenv virtualenv 3.9.12 repeat_identification

poetry install
```

 ## Data Preparation:
The model uses fasta and its paired annotation to make predictions. <br/>
**Download And Generate Required Files:**

Download a genome assembly, initially the human reference genome `GRCh38` (called hg38 by Dfam), from the UCSC Genome Browser.

Download repeat annotations from [Dfam](https://www.dfam.org/home) and generate a subset of the annotations by selecting the desired repeat family or subtype. The selected annotations are saved as the repeat boundaries to create the sequence segmentation dataset.

In order to download and generate needed files should run:
```shell
python generate_dataset.py --species
```
If you use human genome run like: `python generate_dataset.py --species hg38 --configuration configuration.yaml`<br/>
All of the option can add personalized configuration by `configuration.yaml`

And the generated files will seperate by chromosome, and the generated file will looks like:
```shell
>data
>>genome_assemblies
>>>hg38.fa
>>>hg38.fa.gz
>>>datasets
chr1.fa
chr1.fa.fai
…………
>>annotations
hg38_chr10.pickle
>>>
hg38.hits
hg38.hits.gz
hg38_chr14.csv
hg38_chr7.csv 
…………
repeat_families.json
```

<br/>


**Configuration Parameter:**<br/>
Some configuration should be defined in this stage.
```shell
# experiment files directory
save_directory: experiments

# experiment naming prefix
experiment_prefix: standard
################################################################################

# dataset
################################################################################
chromosomes: ['chr1']
dataset_id: hg38
#chromosomes: ['chr10']
segment_length: 2000
overlap: 500
num_queries: 10
#repeat_types: ["LTR", "DNA", "LINE", "SINE", "RC", "Retroposon", "PLE", "Satellite", "tRNA", "snRNA", "rRNA", "scRNA"]
repeat_types: ["DNA"]
dataset_size: 10
################################################################################

```
Additionally, the length of subsequence can be defined by user, but it would not large than 4000.
<br/>

## Train the Model:
 You are gonna train your own model in `model.py` script..<br/>
To train the model, run:<br/>
```shell
python train.py --configuration configuration.yaml
```

**Configuration Parameter:**<br/>
Additionally, some configuration should be defined in this stage.
```shell
# features
################################################################################
cost_class: 1
cost_segments: 1
cost_siou: 1
eos_coef: 1
iou_threshold: 0.5
################################################################################

# network architecture
################################################################################
################################################################################
embedding_dimension: 6
nhead: 6
num_encoder_layers: 1
num_decoder_layers: 1
# training
################################################################################
lr: 0.0001
max_epochs: 1
batch_size: 2
validation_ratio: 0.1
test_ratio: 0.1
max_norm: 0
seed: 42
dropout: 0.3
gpus: 0
loss_delta: 0
patience: 5
profiler: null
num_sample_predictions: 5
################################################################################

```
<br/>

## The visualized result of DETR prediction:
<br/>
Visualizing the predictions of the network will help us understand them better and debug and finetune the model. The result will looks like:
<br/>
```
ground truth: TCCCTCCCTCCTTC444444444444444444444444444444444CTCAGCAGTCGCT
prediction: TCCCTCCCT444444444444444444444444445555555CATTCCTCAGCAGTCGCT
```
<br/>

## Further step
We also provide the alternative way to identity repeat region, in here the transformer model is used, due to time limit, there is no much results here. This project is still under development.<br/>
**Alternative model architecture**
![](fig/newnetwork.gif)
**The visualized result of prediction**
This model is simple than DETR, but it also can get the repeat region. The visualized output will looks like:
```
**********🍓🍓🍓🍓🍓🍓*********🍓🍓🍓🍓🍓🍓*****
```
Each '*' represent the base is not repeat region, each '🍓' represent the repeat region include its type.

## Research papers / References
#### Some of the papers which have been published in recognizing repeat sequence: <br/>
1. [Automated De Novo Identification of Repeat Sequence Families in Sequenced Genomes](https://genome.cshlp.org/content/12/8/1269.short)<br/>

2. [Fast and global detection of periodic sequence repeats in large genomic resources](https://academic.oup.com/nar/article/47/2/e8/5124599?login=false)<br/>

3. [Patterns of de novo tandem repeat mutations and their role in autism](https://www.nature.com/articles/s41586-020-03078-7)<br/>

