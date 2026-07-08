from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_ROOT / "results" / "distilbert_calibration_outputs"
TEST_FILE = INPUT_DIR / "distilbert_test_predictions.csv"
TEST_SCALED_FILE = INPUT_DIR / "distilbert_test_predictions_scaled.csv"
PLOTS_DIR = INPUT_DIR / "plots"
RELIABILITY_PLOT = PLOTS_DIR / "distilbert_reliability_diagram.png"
CONFIDENCE_PLOT = PLOTS_DIR / "distilbert_confidence_histogram.png"


def read_prediction_csv(path: Path) -> Tuple[List[int], List[int], List[float], List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    labels: List[int] = []
    preds: List[int] = []
    confidences: List[float] = []
    probs_pos: List[float] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"label", "pred_label", "confidence", "prob_1"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

        for row in reader:
            labels.append(int(row["label"]))
            preds.append(int(row["pred_label"]))
            confidences.append(float(row["confidence"]))
            probs_pos.append(float(row["prob_1"]))

    return labels, preds, confidences, probs_pos


def compute_reliability_bins(
    labels: List[int],
    preds: List[int],
    confidences: List[float],
    bins: int = 10,
) -> Tuple[List[float], List[float], List[int]]:
    bin_totals = [0] * bins
    bin_correct = [0] * bins
    bin_conf_sum = [0.0] * bins

    for y, p, conf in zip(labels, preds, confidences):
        idx = min(int(conf * bins), bins - 1)
        bin_totals[idx] += 1
        bin_correct[idx] += int(y == p)
        bin_conf_sum[idx] += conf

    avg_accs: List[float] = []
    avg_confs: List[float] = []
    totals: List[int] = []
    for total, correct, conf_sum in zip(bin_totals, bin_correct, bin_conf_sum):
        if total == 0:
            avg_accs.append(0.0)
            avg_confs.append(0.0)
            totals.append(0)
        else:
            avg_accs.append(correct / total)
            avg_confs.append(conf_sum / total)
            totals.append(total)

    return avg_confs, avg_accs, totals


def plot_reliability_diagram(
    before,
    after,
    output_path: Path,
    bins: int = 10,
    *,
    model_name: str = "DistilBERT",
    dataset_name: str = "EduPhish",
    min_bin_count: int = 5,
) -> None:
    before_labels, before_preds, before_confs = before
    after_labels, after_preds, after_confs = after

    before_avg_confs, before_avg_accs, before_totals = compute_reliability_bins(
        before_labels, before_preds, before_confs, bins=bins
    )
    after_avg_confs, after_avg_accs, after_totals = compute_reliability_bins(
        after_labels, after_preds, after_confs, bins=bins
    )

    bin_centers = [(i + 0.5) / bins for i in range(bins)]
    width = 1.0 / bins * 0.8

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for ax, title, avg_confs, avg_accs, totals in [
        (axes[0], "Before Temperature Scaling", before_avg_confs, before_avg_accs, before_totals),
        (axes[1], "After Temperature Scaling", after_avg_confs, after_avg_accs, after_totals),
    ]:
        visible_points = [
            (center, conf, acc)
            for center, conf, acc, total in zip(bin_centers, avg_confs, avg_accs, totals)
            if total >= min_bin_count
        ]
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
        if visible_points:
            xs = [center for center, _, _ in visible_points]
            confs = [conf for _, conf, _ in visible_points]
            accs = [acc for _, _, acc in visible_points]
            ax.bar(xs, accs, width=width, align="center", alpha=0.55, edgecolor="black", label=title)
            ax.plot(confs, accs, marker="o", color="#1f77b4", linewidth=1.6, label=title)
        ax.set_title(title)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.suptitle(
        f"Reliability Diagram Before and After Temperature Scaling ({model_name} on {dataset_name})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_histogram(
    before_confs: List[float],
    after_confs: List[float],
    output_path: Path,
    bins: int = 10,
    *,
    model_name: str = "DistilBERT",
    dataset_name: str = "EduPhish",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(before_confs, bins=bins, alpha=0.5, label="Before Temperature Scaling", color="#1f77b4", edgecolor="white")
    ax.hist(after_confs, bins=bins, alpha=0.5, label="After Temperature Scaling", color="#ff7f0e", edgecolor="white")
    ax.set_title(f"Confidence Histogram ({model_name} on {dataset_name})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Prediction Count")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    before_labels, before_preds, before_confidences, _ = read_prediction_csv(TEST_FILE)
    after_labels, after_preds, after_confidences, _ = read_prediction_csv(TEST_SCALED_FILE)

    if before_labels != after_labels:
        raise ValueError("The before/after files do not contain matching labels/rows.")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_reliability_diagram(
        (before_labels, before_preds, before_confidences),
        (after_labels, after_preds, after_confidences),
        RELIABILITY_PLOT,
        model_name="DistilBERT",
        dataset_name="EduPhish",
        min_bin_count=5,
    )
    plot_confidence_histogram(
        before_confidences,
        after_confidences,
        CONFIDENCE_PLOT,
        model_name="DistilBERT",
        dataset_name="EduPhish",
    )

    print(f"Saved reliability diagram -> {RELIABILITY_PLOT}")
    print(f"Saved confidence histogram -> {CONFIDENCE_PLOT}")


if __name__ == "__main__":
    main()
