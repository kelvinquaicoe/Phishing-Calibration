import subprocess
from pathlib import Path

PYTHON = "python3"

validation_file = "data/distilbert_eduphish_validation.csv"

models = {
    "bert": ("results/bert_eduphish/checkpoint-500", "bert-base-uncased"),
    "distilbert": ("results/distilbert_eduphish/checkpoint-1695", "distilbert-base-uncased"),
    "roberta": ("results/roberta_eduphish/checkpoint-500", "roberta-base"),
    "albert": ("results/albert_phishing_email/checkpoint-500", "albert-base-v2"),
    "deberta": ("results/deberta_eduphish/checkpoint-500", "microsoft/deberta-v3-small"),
    "deberta_v3_base": ("results/deberta_v3_base_phishing_email/checkpoint-500", "microsoft/deberta-v3-base"),
}

datasets = {
    "ceas08": "data/eval_subsets/ceas08_eval_5000.csv",
    "enron": "data/eval_subsets/enron_eval_5000.csv",
    "spamassassin": "data/eval_subsets/spamassassin_eval_5000.csv",
}

for model_name, (model_dir, tokenizer_name) in models.items():
    for dataset_name, test_file in datasets.items():
        output_dir = f"results/{model_name}_{dataset_name}_calibration_outputs"

        print("=" * 80)
        print(f"Running {model_name} on {dataset_name}")
        print("=" * 80)

        cmd = [
            PYTHON,
            "src/run_calibration.py",
            "--model-dir", model_dir,
            "--validation-file", validation_file,
            "--test-file", test_file,
            "--output-dir", output_dir,
            "--prefix", model_name,
            "--tokenizer-name", tokenizer_name,
        ]

        subprocess.run(cmd, check=True)
