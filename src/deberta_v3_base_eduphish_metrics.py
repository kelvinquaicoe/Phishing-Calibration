from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "deberta_v3_base_calibration_outputs"


def _raise_csv_field_limit() -> None:
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size = int(max_size / 10)


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _softmax2(logit_0: float, logit_1: float) -> tuple[float, float]:
    # stable softmax for 2 classes
    m = max(logit_0, logit_1)
    e0 = math.exp(logit_0 - m)
    e1 = math.exp(logit_1 - m)
    z = e0 + e1
    return (e0 / z, e1 / z)


@dataclass
class CalibrationBin:
    count: int = 0
    conf_sum: float = 0.0
    correct_sum: float = 0.0


@dataclass
class MetricsResult:
    path: str
    n: int
    accuracy: float
    nll: float
    brier: float
    ece: float
    mce: float
    positive_rate: float
    pred_positive_rate: float
    confusion_matrix: dict[str, int]
    num_bins: int


def _bin_index(conf: float, num_bins: int) -> int:
    if not (0.0 <= conf <= 1.0) or math.isnan(conf):
        return -1
    # Put 1.0 in the last bin.
    idx = min(int(conf * num_bins), num_bins - 1)
    return idx


def compute_metrics(predictions_csv: Path, *, num_bins: int = 15) -> tuple[MetricsResult, list[dict[str, float]]]:
    _raise_csv_field_limit()

    bins = [CalibrationBin() for _ in range(num_bins)]
    skipped = 0

    n = 0
    correct = 0
    nll_sum = 0.0
    brier_sum = 0.0
    label_pos = 0
    pred_pos = 0
    tp = fp = tn = fn = 0

    with predictions_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty CSV: {predictions_csv}")

        col = {name: i for i, name in enumerate(header)}
        required_any = {"prob_0", "prob_1", "logit_0", "logit_1"}
        if "label" not in col:
            raise ValueError(f"Missing required column 'label' in {predictions_csv}")
        if not (required_any & set(col)):
            raise ValueError(
                f"Expected prob/logit columns in {predictions_csv}. Found: {', '.join(header)}"
            )

        for row in reader:
            if not row:
                continue

            try:
                y = int(row[col["label"]])
            except Exception:
                skipped += 1
                continue

            prob_0: Optional[float] = None
            prob_1: Optional[float] = None
            if "prob_0" in col and "prob_1" in col:
                prob_0 = _safe_float(row[col["prob_0"]])
                prob_1 = _safe_float(row[col["prob_1"]])
            elif "logit_0" in col and "logit_1" in col:
                logit_0 = _safe_float(row[col["logit_0"]])
                logit_1 = _safe_float(row[col["logit_1"]])
                prob_0, prob_1 = _softmax2(logit_0, logit_1)

            if prob_0 is None or prob_1 is None or math.isnan(prob_0) or math.isnan(prob_1):
                skipped += 1
                continue

            # Normalize (defensive) and clamp.
            s = prob_0 + prob_1
            if s <= 0 or math.isnan(s):
                skipped += 1
                continue
            prob_0 /= s
            prob_1 /= s
            prob_0 = min(max(prob_0, 0.0), 1.0)
            prob_1 = min(max(prob_1, 0.0), 1.0)

            pred = 1 if prob_1 >= prob_0 else 0
            conf = max(prob_0, prob_1)
            is_correct = 1.0 if pred == y else 0.0

            # Aggregate core metrics.
            n += 1
            correct += int(is_correct)
            label_pos += int(y == 1)
            pred_pos += int(pred == 1)

            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 0:
                tn += 1
            else:
                fn += 1

            p_true = prob_1 if y == 1 else prob_0
            p_true = min(max(p_true, 1e-15), 1.0)
            nll_sum += -math.log(p_true)
            brier_sum += (prob_1 - float(y)) ** 2

            b = _bin_index(conf, num_bins)
            if b >= 0:
                bins[b].count += 1
                bins[b].conf_sum += conf
                bins[b].correct_sum += is_correct

    if n == 0:
        raise ValueError(f"No valid rows found in {predictions_csv} (skipped={skipped})")

    ece = 0.0
    mce = 0.0
    reliability_rows: list[dict[str, float]] = []

    for b, bin_stats in enumerate(bins):
        if bin_stats.count == 0:
            reliability_rows.append(
                {
                    "bin": float(b),
                    "count": 0.0,
                    "confidence_mean": float("nan"),
                    "accuracy_mean": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue

        conf_mean = bin_stats.conf_sum / bin_stats.count
        acc_mean = bin_stats.correct_sum / bin_stats.count
        gap = abs(acc_mean - conf_mean)
        weight = bin_stats.count / n

        ece += weight * gap
        mce = max(mce, gap)

        reliability_rows.append(
            {
                "bin": float(b),
                "count": float(bin_stats.count),
                "confidence_mean": float(conf_mean),
                "accuracy_mean": float(acc_mean),
                "gap": float(gap),
            }
        )

    result = MetricsResult(
        path=str(predictions_csv),
        n=n,
        accuracy=correct / n,
        nll=nll_sum / n,
        brier=brier_sum / n,
        ece=ece,
        mce=mce,
        positive_rate=label_pos / n,
        pred_positive_rate=pred_pos / n,
        confusion_matrix={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        num_bins=num_bins,
    )

    return result, reliability_rows


def _default_inputs(output_dir: Path) -> list[Path]:
    candidates = [
        output_dir / "deberta_v3_base_test_predictions.csv",
        output_dir / "deberta_v3_base_test_predictions_scaled.csv",
        output_dir / "deberta_v3_base_validation_predictions.csv",
    ]
    return [p for p in candidates if p.exists()]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_reliability_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
    rows = list(rows)
    fieldnames = ["bin", "count", "confidence_mean", "accuracy_mean", "gap"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute accuracy + calibration metrics from BERT prediction CSVs.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Prediction CSV(s) to evaluate. Defaults to files under results/bert_calibration_outputs/.",
    )
    parser.add_argument("--bins", type=int, default=15, help="Number of ECE bins (default: 15).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write metrics JSON + reliability CSV (default: results/bert_calibration_outputs/).",
    )

    args = parser.parse_args()

    inputs = [p for p in args.inputs] if args.inputs else _default_inputs(args.output_dir)
    if not inputs:
        raise SystemExit(
            f"No inputs provided and no default files found under: {args.output_dir}. "
            "Run calibration first, or pass explicit CSV paths."
        )

    for p in inputs:
        metrics, reliability = compute_metrics(p, num_bins=args.bins)
        stem = Path(metrics.path).stem
        metrics_path = args.output_dir / f"{stem}_metrics.json"
        reliability_path = args.output_dir / f"{stem}_reliability.csv"

        _write_json(metrics_path, metrics.__dict__)
        _write_reliability_csv(reliability_path, reliability)

        print(f"\n== {p.name} ==")
        print(f"n: {metrics.n}")
        print(f"accuracy: {metrics.accuracy:.6f}")
        print(f"nll: {metrics.nll:.6f}")
        print(f"brier: {metrics.brier:.6f}")
        print(f"ece({metrics.num_bins}): {metrics.ece:.6f}")
        print(f"mce({metrics.num_bins}): {metrics.mce:.6f}")
        print(f"positive_rate: {metrics.positive_rate:.6f}")
        print(f"pred_positive_rate: {metrics.pred_positive_rate:.6f}")
        print(f"confusion_matrix: {metrics.confusion_matrix}")
        print(f"wrote: {metrics_path}")
        print(f"wrote: {reliability_path}")


if __name__ == "__main__":
    main()

