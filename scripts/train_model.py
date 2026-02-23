import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Bio import SeqIO

# Load the balanced dataset
df = pd.read_csv("data/balanced_dataset.csv")

# Precompute all possible 3-mers
BASES = ["A", "C", "G", "T"]
ALL_KMERS = [a + b + c for a in BASES for b in BASES for c in BASES]

def get_kmer_frequencies(sequence):
    """
    Convert a DNA sequence into normalized 3-mer frequencies.
    """
    kmer_dict = {kmer: 0 for kmer in ALL_KMERS}

    for i in range(len(sequence) - 3 + 1):
        kmer = sequence[i:i+3]
        if kmer in kmer_dict:
            kmer_dict[kmer] += 1

    counts = np.array(list(kmer_dict.values()))
    return counts / counts.sum()

# Transform sequences into a feature matrix of 3-mer frequencies
x = np.array([get_kmer_frequencies(seq) for seq in df["sequence"]])
y = df["label"].values

# Train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=24)
model.fit(x, y)

# Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/clade_predictor.pkl")
