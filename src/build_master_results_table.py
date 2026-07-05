import json
from pathlib import Path
import pandas as pd

models = ["bert", "distilbert", "roberta", "albert", "deberta", "deberta_v3_base"]

datasets = {
    "eduphish": {
        "bert": "results/bert_calibration_outputs/bert",
        "distilbert": "results/distilbert_calibration_outputs/distilbert",
        "roberta": "results/roberta_calibration_outputs/roberta",
        "albert": "results/albert_calibration_outputs/albert",
        "deberta": "results/deberta_calibration_outputs/deberta",
        "deberta_v3_base": "results/deberta_v3_base_calibration_outputs/deberta_v3_base",
    },
    "ceas08": None,
    "enron": None,
    "spamassassin": None,
}

rows = []

for dataset, special_paths in datasets.items():
    for model in models:
        if dataset == "eduphish":
            stem = Path(special_paths[model])
        else:
            stem = Path(f"results/{model}_{dataset}_calibration_outputs/{model}")

        before_path = Path(f"{stem}_test_predictions_metrics.json")
        after_path = Path(f"{stem}_test_predictions_scaled_metrics.json")

        if not before_path.exists() or not after_path.exists():
            print(f"Missing: {model} {dataset}")
            continue

        before = json.loads(before_path.read_text())
        after = json.loads(after_path.read_text())

        rows.append({
            "model": model,
            "dataset": dataset,
            "accuracy": before["accuracy"],
            "nll_before": before["nll"],
            "nll_after": after["nll"],
            "ece_before": before["ece"],
            "ece_after": after["ece"],
            "mce_before": before["mce"],
            "mce_after": after["mce"],
            "brier_before": before["brier"],
            "brier_after": after["brier"],
        })

df = pd.DataFrame(rows)

out_csv = Path("results/master_results_table.csv")
out_md = Path("results/master_results_table.md")

df.to_csv(out_csv, index=False)

pretty = df.copy()
for col in pretty.columns:
    if col not in ["model", "dataset"]:
        pretty[col] = pretty[col].map(lambda x: f"{x:.4f}")

out_md.write_text(pretty.to_markdown(index=False))

print(f"Wrote {out_csv}")
print(f"Wrote {out_md}")
print()
print(pretty.to_markdown(index=False))
