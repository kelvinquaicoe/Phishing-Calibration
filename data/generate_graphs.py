from __future__ import annotations

import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent
GRAPH_DIR = ROOT / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

DATASETS = {
    "EduPhish": ROOT / "Education-Targeted Phishing Email Dataset" / "eduphish_dataset.csv",
    "EduPhish Train": ROOT / "train_eduphish.csv",
    "EduPhish Test": ROOT / "test_eduphish.csv",
    "DistilBERT Train": ROOT / "distilbert_eduphish_train.csv",
    "DistilBERT Validation": ROOT / "distilbert_eduphish_validation.csv",
    "Phishing Email Train": ROOT / "train_phishing_email.csv",
    "Phishing Email Test": ROOT / "test_phishing_email.csv",
    "Phishing-Legit Train": ROOT / "train_phishing_legit.csv",
    "Phishing-Legit Test": ROOT / "test_phishing_legit.csv",
    "KD 10k": ROOT / "phishing_legit_dataset_KD_10000.csv",
}


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        yield from reader


def count_labels(path: Path):
    counts = Counter()
    total = 0
    for row in read_rows(path):
        total += 1
        label = row.get("label", "unknown")
        counts[label] += 1
    return total, counts


def text_length(text: str) -> int:
    return len((text or "").split())


def save_bar(labels, values, title, ylabel, filename, color="#4C78A8"):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=200)
    plt.close()


# 1) Dataset sizes
sizes = []
for name, path in DATASETS.items():
    if path.exists():
        total, _ = count_labels(path)
        sizes.append((name, total))

sizes.sort(key=lambda x: x[1], reverse=True)
save_bar(
    [x[0] for x in sizes],
    [x[1] for x in sizes],
    "Dataset Sizes",
    "Number of samples",
    "dataset_sizes.png",
    color="#72B7B2",
)

# 2) EduPhish class distribution
for dataset_name, file_name in [("EduPhish", "eduphish_dataset.csv"), ("EduPhish Train", "train_eduphish.csv"), ("EduPhish Test", "test_eduphish.csv")]:
    path = ROOT / ("Education-Targeted Phishing Email Dataset" if dataset_name == "EduPhish" else "") / file_name if dataset_name == "EduPhish" else ROOT / file_name
    if not path.exists():
        continue
    _, counts = count_labels(path)
    labels = ["Legitimate (0)", "Phishing (1)"]
    values = [counts.get("0", 0), counts.get("1", 0)]
    safe_name = dataset_name.lower().replace(" ", "_") + "_class_distribution.png"
    save_bar(labels, values, f"{dataset_name} Class Distribution", "Count", safe_name, color="#F58518")

# 3) KD dataset categorical distributions
kd_path = ROOT / "phishing_legit_dataset_KD_10000.csv"
if kd_path.exists():
    ph_type_counter = Counter()
    severity_counter = Counter()
    confidences = []
    text_lengths = {"0": [], "1": []}
    for row in read_rows(kd_path):
        ph_type_counter[row.get("phishing_type", "unknown")] += 1
        severity_counter[row.get("severity", "unknown")] += 1
        conf = row.get("confidence")
        try:
            confidences.append(float(conf))
        except (TypeError, ValueError):
            pass
        label = row.get("label", "unknown")
        text_lengths[label].append(text_length(row.get("text", "")))

    save_bar(
        [k.replace("_", " ") for k in ph_type_counter.keys()],
        list(ph_type_counter.values()),
        "KD Dataset: Phishing Type Distribution",
        "Count",
        "kd_phishing_type_distribution.png",
        color="#54A24B",
    )

    save_bar(
        list(severity_counter.keys()),
        list(severity_counter.values()),
        "KD Dataset: Severity Distribution",
        "Count",
        "kd_severity_distribution.png",
        color="#E45756",
    )

    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, color="#B279A2", edgecolor="white")
    plt.title("KD Dataset: Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "kd_confidence_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(text_lengths.get("0", []), bins=30, alpha=0.7, label="Legitimate (0)", color="#72B7B2")
    plt.hist(text_lengths.get("1", []), bins=30, alpha=0.7, label="Phishing (1)", color="#F58518")
    plt.title("KD Dataset: Text Length by Class")
    plt.xlabel("Token count")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "kd_text_length_by_class.png", dpi=200)
    plt.close()

print(f"Graphs saved to: {GRAPH_DIR}")
