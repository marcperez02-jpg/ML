import matplotlib.pyplot as plt
import numpy as np
import joblib

REPORT_PATH = "models/validation_report.txt"
OUT_METRICS = "images/performance_metrics.png"
OUT_MATRIX = "images/confusion_matrix.png"
MODEL_PATH = "models/clade_predictor.pkl"
OUT_FEATURES = "images/top_kmers.png"

# Read the report values and store them in a dictionary
values = {}
with open(REPORT_PATH, "r") as handle:
    for line in handle:
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()

# Get metrics and counts from the validation report
precision = float(values["Precision"])
recall = float(values["Recall"])
tp = int(values["True Positives"])
tn = int(values["True Negatives"])
fp = int(values["False Positives"])
fn = int(values["False Negatives"])

# Plot precision and recall into a bar chart
plt.figure(figsize=(8, 6))
plt.bar(["Precision", "Recall"], [precision, recall], color=["green", "blue"])
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Model Performance")
plt.savefig(OUT_METRICS)
print(f"Saved {OUT_METRICS}")

# Plot the confusion matrix into a heatmap
cm = np.array([[tn, fp], [fn, tp]])
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["Other", "2.3.4.4b"])
plt.yticks([0, 1], ["Other", "2.3.4.4b"])

# Write the numeric value inside each heatmap cell for readability
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.savefig(OUT_MATRIX)
print(f"Saved {OUT_MATRIX}")

# Load the trained model
model = joblib.load(MODEL_PATH)

# Read feature importances from the file
feature_importances_path = "models/feature_importances.txt"
kmer_importances = []
with open(feature_importances_path, "r") as f:
    next(f)  # Skip the header
    for line in f:
        kmer, importance = line.strip().split("\t")
        kmer_importances.append((kmer, float(importance)))

# Sort by importance and select the top 10
kmer_importances = sorted(kmer_importances, key=lambda x: x[1], reverse=True)
top_kmers = kmer_importances[:10]

# Plot the top 10 3-mers and their importances
kmers, importances = zip(*top_kmers)
plt.figure(figsize=(10, 6))
plt.bar(kmers, importances, color="purple")
plt.ylabel("Importance")
plt.xlabel("3-mers")
plt.title("Top 10 Most Influential 3-mers")
plt.xticks(rotation=45, ha="right")
plt.savefig(OUT_FEATURES)
print(f"Saved {OUT_FEATURES}")