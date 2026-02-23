#!/usr/bin/env python3

import pandas as pd
from Bio import SeqIO

def process_fasta(file_path, label):
    """
    Load sequences from a FASTA file, apply cleaning filters, 
    and return a list of dicts with sequences and labels.
    """
    data = []
    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq).upper()

        # Keep sequences within a reasonable HA length range
        if len(seq) < 1600 or len(seq) > 1900:
            continue

        # Drop sequences with too many Ns
        if seq.count("N") / len(seq) > 0.01:
            continue

        data.append({"sequence": seq, "label": label})
    return data

def filter_overlapping_sequences(df, test_set_path):
    """
    Remove sequences from the dataset that overlap with the test set.
    """
    # Load test sequences
    test_sequences = set()
    for record in SeqIO.parse(test_set_path, "fasta"):
        test_sequences.add(str(record.seq).upper())

    # Filter out overlapping sequences
    return df[~df["sequence"].isin(test_sequences)]

# Load and clean both datasets
pos_data = process_fasta("data/positive.fasta", 1)
neg_data = process_fasta("data/negative.fasta", 0)

df_raw = pd.DataFrame(pos_data + neg_data)

# Filter out sequences overlapping with the test set
test_set_path = "data/test/test.fasta"
df_raw = filter_overlapping_sequences(df_raw, test_set_path)

# Balance classes by downsampling both classes to the smaller count
pos_count = len(df_raw[df_raw["label"] == 1])
neg_count = len(df_raw[df_raw["label"] == 0])
minority_count = min(pos_count, neg_count)
df_positive = df_raw[df_raw["label"] == 1]
df_negative = df_raw[df_raw["label"] == 0]
# Randomly keep only as many samples as the smaller class has
df_positive_balanced = df_positive.sample(n=minority_count, random_state=24)
df_negative_balanced = df_negative.sample(n=minority_count, random_state=24)
# Merge the balanced classes, shuffle, and reset the index
df = pd.concat([df_positive_balanced, df_negative_balanced]).sample(frac=1, random_state=24).reset_index(drop=True)

# Save the balanced dataset
df.to_csv("data/balanced_dataset.csv", index=False)
