# H5N1 Clade 2.3.4.4b Predictor

[cite_start]This repository contains an automated machine learning pipeline designed to identify whether a Hemagglutinin (HA) sequence from an influenza A virus belongs to the highly pathogenic **H5N1 clade 2.3.4.4b**[cite: 157, 162].

## Overview
[cite_start]The 2.3.4.4b lineage has shown significant pandemic potential due to high virulence and increased zoonotic transmission[cite: 158]. [cite_start]This tool uses genomic "fingerprints" (3-mer motifs) to classify sequences with high precision[cite: 201, 269].

## Pipeline Structure
[cite_start]The workflow is managed by a **Snakefile**, which orchestrates the following stages[cite: 163, 223]:

1.  [cite_start]**Test Set Construction**: Randomly samples 50 positive and 50 negative sequences to create a balanced, independent 100-sequence test set[cite: 168, 173].
2.  [cite_start]**Data Cleaning**: Filters for full-length HA sequences (1600–1900 bp) and removes sequences with >1% ambiguous "N" bases[cite: 177].
3.  [cite_start]**Leakage Prevention**: Explicitly removes any sequence present in the test set from the training dataset[cite: 179].
4.  [cite_start]**Training**: Implements a **Random Forest Classifier** using k-mer featurization (3-mers)[cite: 192, 196].
5.  [cite_start]**Validation**: Evaluates the model using Precision, Recall, and a Confusion Matrix[cite: 214, 215].
6.  [cite_start]**Visualization**: Generates performance metrics and feature importance plots[cite: 218].


## Requirements
* Python 3.x
* Biopython
* Scikit-learn
* Pandas/NumPy
* Snakemake
* Joblib
* Matplotlib

## Usage
[cite_start]To initialize the full workflow (data cleaning, training, and evaluation), run[cite: 224]:
```bash
snakemake --cores 1