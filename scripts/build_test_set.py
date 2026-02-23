#!/usr/bin/env python3

import os
import random
from Bio import SeqIO

# Load sequences from the source datasets (Positive = clade 2.3.4.4b, Negative = other clades)
pos_records = list(SeqIO.parse("data/positive.fasta", "fasta"))
neg_records = list(SeqIO.parse("data/negative.fasta", "fasta"))

if len(pos_records) < 50 or len(neg_records) < 50:
    raise ValueError("Need at least 50 sequences in each dataset")

# Sample a fixed number of sequences from each set randomly, using a fixed seed for reproducibility.
rng = random.Random(24)
pos_sample = rng.sample(pos_records, 50)
neg_sample = rng.sample(neg_records, 50)

# Add labels to the FASTA headers for downstream evaluation
for record in pos_sample:
    record.description = f"{record.id} label=1"

for record in neg_sample:
    record.description = f"{record.id} label=0"

# Write the combined test set to a new FASTA file stored in the data/test directory
os.makedirs("data/test", exist_ok=True)
with open("data/test/test.fasta", "w") as handle:
    SeqIO.write(pos_sample + neg_sample, handle, "fasta")


