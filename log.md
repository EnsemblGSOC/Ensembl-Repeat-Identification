# mark some pins.

## scripts

1. *retrieve_genome_assembly.py*

Download a genome assembly, initially the human reference genome GRCh38 (called hg38 by Dfam), from the UCSC Genome Browser.

Split the genome sequences to smaller files ?

The data will look like the following:
```
sequence format: chr1: 1 - 100000, NNNNNN
```

2. *retrieve_annotation.py*

Download repeat annotations from [Dfam](https://www.dfam.org/home) and generate a subset of the annotations by selecting the desired repeat family or subtype. The selected annotations are saved as the repeat boundaries to create the sequence segmentation dataset.

retrieve_annotation.py
```
label format: chr1 start end subtype
```

3. *generate_dataset.py*

Dataset generation script with arguments for different dataset generation subtasks.
We first choose 'LTR family' as training datasets, the length of 'LTR family' have variant length from 100bp to 5kb. So, when we training the datasets, if the result do not work well, this is one of the reason, we should consider.

4. *utils.py*

Project library module.
