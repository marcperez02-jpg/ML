import os
import joblib
import numpy as np
from Bio import SeqIO
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from train_model import get_kmer_frequencies

# Build a list of all possible 3-letter DNA combinations
BASES = ["A", "C", "G", "T"]
ALL_KMERS = [a + b + c for a in BASES for b in BASES for c in BASES]

# Paths for the saved model and the test FASTA
model_path = "models/clade_predictor.pkl"
test_file = "data/test/test.fasta"

def get_label_from_description(description):
    """Read label=0/1 from a FASTA header."""
    for part in description.split():
        if part.startswith("label="):
            value = part.split("=", 1)[1]
            if value in {"0", "1"}:
                return int(value)
    raise ValueError("Missing label=0/1 in FASTA description")

# Load the trained model
model = joblib.load(model_path)

# Store true labels and model predictions
y_true = []
y_pred = []

# Read all test sequences from FASTA
records = list(SeqIO.parse(test_file, "fasta"))

for record in records:
    # Convert the sequence to uppercase text
    test_seq = str(record.seq).upper()
    # Turn the sequence into numeric features for the model
    features = get_kmer_frequencies(test_seq)
    # Ensure features array is 2D
    features = features.reshape(1, -1)
    # model.predict comes from the trained RandomForest loaded above
    prediction = model.predict(features)[0]
    # Read the true label from the FASTA header
    true_label = get_label_from_description(record.description)

    y_true.append(true_label)
    y_pred.append(prediction)

# Calculate summary metrics from predictions
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()


# Write a short report to a text file
report_path = "models/validation_report.txt"

with open(report_path, "w") as f:
    f.write("H5N1 CLADE PREDICTOR VALIDATION REPORT\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"True Positives:  {tp}\n")
    f.write(f"True Negatives:  {tn}\n")
    f.write(f"False Positives: {fp}\n")
    f.write(f"False Negatives: {fn}\n")
    

print(f"Report saved to {report_path}")

# Calculate feature importances and save them to a file
feature_importances = model.feature_importances_
feature_importances_path = "models/feature_importances.txt"

with open(feature_importances_path, "w") as f:
    f.write("3-mer\tImportance\n")
    for kmer, importance in zip(ALL_KMERS, feature_importances):
        f.write(f"{kmer}\t{importance:.6f}\n")

print(f"Feature importances saved to {feature_importances_path}")