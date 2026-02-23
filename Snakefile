# -*- snakemake-*-
# Configuration: Define the paths to your input data
DATA_POS = "data/positive.fasta"
DATA_NEG = "data/negative.fasta"
TEST_FILE = "data/test/test.fasta"

# The 'all' rule defines the final targets of the entire pipeline
rule all:
    input:
        "models/clade_predictor.pkl",
        "models/validation_report.txt",
        "images/performance_metrics.png",
        "images/confusion_matrix.png"

# Rule 0: Build a test set from the positive and negative datasets
rule build_test_set:
    input:
        pos = DATA_POS,
        neg = DATA_NEG
    output:
        test = TEST_FILE
    shell:
        "python scripts/build_test_set.py"

# Rule 1: Clean data and create the balanced dataset
rule dataclean:
    input:
        pos = DATA_POS,
        neg = DATA_NEG,
        test = TEST_FILE  # Ensure test set is created first
    output:
        dataset = "data/balanced_dataset.csv"
    shell:
        "python scripts/data_cleaning.py"

# Rule 2: Train the Random Forest model
rule train:
    input:
        dataset = "data/balanced_dataset.csv"
    output:
        model = "models/clade_predictor.pkl"
    shell:
        "python scripts/train_model.py"

# Rule 3: Validate the model and generate the text report
rule validate:
    input:
        model = "models/clade_predictor.pkl",
        test = TEST_FILE
    output:
        report = "models/validation_report.txt"
    shell:
        "python scripts/validate_predictor.py"

# Rule 4: Generate the visual charts for GitHub
rule visualize:
    input:
        # This ensures visualization only happens after a successful validation
        report = "models/validation_report.txt"
    output:
        metrics = "images/performance_metrics.png",
        matrix = "images/confusion_matrix.png"
    shell:
        "python scripts/visualize.py"

# Rule 5: Clean up generated files (Optional but very helpful)
rule clean:
    shell:
        "rm -f models/*.pkl models/*.txt images/*.png data/balanced_dataset.csv data/test/test.fasta"