import subprocess
from pathlib import Path

PYTHON = "python3"

models = [
    "bert",
    "distilbert",
    "roberta",
    "albert",
    "deberta",
    "deberta_v3_base",
]

datasets = [
    "ceas08",
    "enron",
    "spamassassin",
]

for model in models:
    for dataset in datasets:
        output_dir = Path(f"results/{model}_{dataset}_calibration_outputs")

        print("=" * 80)
        print(f"Computing metrics for {model} on {dataset}")
        print("=" * 80)

        before = output_dir / f"{model}_test_predictions.csv"
        after = output_dir / f"{model}_test_predictions_scaled.csv"

        if not before.exists() or not after.exists():
            print(f"Skipping missing files in {output_dir}")
            continue

        cmd = [
            PYTHON,
            "src/bert_eduphish_metrics.py",
            "--output-dir",
            str(output_dir),
            str(before),
            str(after),
        ]

        subprocess.run(cmd, check=True)
