from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "graphs" / "poster_figures"
PREFERRED_DATASET_ORDER = ["eduphish", "ceas_08", "ceas08", "ceas", "enron", "spamassassin"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="ignore") as fh:
        return list(csv.DictReader(fh))


def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def normalize_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if value <= 1.0 else value


def scale_metric_value(value: float | None, metric: str) -> float | None:
    if value is None:
        return None
    if metric == "ece":
        return normalize_percent(value)
    return value


def format_metric_value(value: float | None, metric: str) -> str:
    if value is None:
        return "n/a"
    if metric == "ece":
        return f"{value:.2f}"
    if metric == "brier":
        return f"{value:.3f}"
    return f"{value:.2f}"


def resolve_results_csv(explicit: Path | None, results_dir: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None

    candidates = [
        ROOT / "results" / "master_results_table.csv",
        ROOT.parent / "results" / "master_results_table.csv",
        ROOT / "master_results_table.csv",
        ROOT.parent / "master_results_table.csv",
    ]
    if results_dir is not None:
        candidates.extend([
            results_dir / "master_results_table.csv",
            results_dir / "master_results.csv",
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_reliability_csvs(
    explicit: list[Path] | None,
    results_dir: Path | None,
    rows: list[dict[str, str]],
) -> list[Path]:
    if explicit:
        return [path for path in explicit if path.exists()]

    base_dir = results_dir or (ROOT / "results")
    candidates: list[Path] = []
    seen: set[Path] = set()

    for row in rows:
        model = str(row.get("model", "")).strip()
        dataset = str(row.get("dataset", "")).strip().lower()
        if not model or not dataset:
            continue

        if dataset == "eduphish":
            stem = base_dir / f"{model}_calibration_outputs" / model
        else:
            stem = base_dir / f"{model}_{dataset}_calibration_outputs" / model

        for suffix in ["_test_predictions_reliability.csv", "_test_predictions_scaled_reliability.csv", "_validation_predictions_reliability.csv"]:
            path = Path(f"{stem}{suffix}")
            if path.exists() and path not in seen:
                seen.add(path)
                candidates.append(path)

    return candidates


def dataset_sort_key(name: str) -> tuple[int, str]:
    lower = name.lower()
    for idx, token in enumerate(PREFERRED_DATASET_ORDER):
        if token in lower:
            return idx, lower
    return len(PREFERRED_DATASET_ORDER), lower


def is_metric_table(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    required = {"model", "dataset", "accuracy", "ece_before", "ece_after", "brier_before", "brier_after"}
    return required.issubset(rows[0].keys())


def build_accuracy_series(rows: list[dict[str, str]]) -> tuple[list[str], list[str], dict[str, dict[str, list[float]]]] | None:
    if not is_metric_table(rows):
        return None

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        acc = to_float(row.get("accuracy"))
        if acc is None:
            continue
        dataset = str(row.get("dataset", "")).strip() or "Dataset"
        model = str(row.get("model", "")).strip() or "Model"
        grouped[dataset][model].append(normalize_percent(acc) or 0.0)

    if not grouped:
        return None

    datasets = sorted(grouped.keys(), key=dataset_sort_key)
    models = sorted({model for per_dataset in grouped.values() for model in per_dataset.keys()})
    return datasets, models, grouped


def reliability_condition_from_name(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ["before", "uncalibrated", "pre_temp", "pretemperature", "pre-scaling"]):
        return "before"
    if any(token in lower for token in ["after", "calibrated", "post_temp", "posttemperature", "post-scaling"]):
        return "after"
    return "unknown"


def parse_reliability_rows(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for idx, row in enumerate(rows):
        lowered = {k.lower(): k for k in row}
        conf_key = next((lowered[k] for k in lowered if any(t in k for t in ["conf", "prob", "mean_conf"])), None)
        acc_key = next((lowered[k] for k in lowered if any(t in k for t in ["acc", "accuracy", "empirical", "observed"])), None)
        count_key = next((lowered[k] for k in lowered if any(t in k for t in ["count", "samples", "support", "n"])), None)
        bin_key = next((lowered[k] for k in lowered if any(t in k for t in ["bin", "bucket", "interval", "range", "center", "mid"])), None)
        lo_key = next((lowered[k] for k in lowered if any(t in k for t in ["lower", "left"])), None)
        hi_key = next((lowered[k] for k in lowered if any(t in k for t in ["upper", "right"])), None)

        conf = to_float(row.get(conf_key)) if conf_key else None
        acc = to_float(row.get(acc_key)) if acc_key else None
        count = to_float(row.get(count_key)) if count_key else None
        x = None

        if bin_key:
            raw_bin = str(row.get(bin_key, ""))
            raw_bin = raw_bin.replace("to", ",").replace("-", ",").replace(":", ",")
            nums = [to_float(part) for part in raw_bin.split(",")]
            nums = [n for n in nums if n is not None]
            if len(nums) >= 2:
                x = (nums[0] + nums[1]) / 2.0
            elif len(nums) == 1:
                x = nums[0]

        if x is None and lo_key and hi_key:
            lo = to_float(row.get(lo_key))
            hi = to_float(row.get(hi_key))
            if lo is not None and hi is not None:
                x = (lo + hi) / 2.0

        if x is None:
            x = conf if conf is not None else idx / max(1, len(rows) - 1)

        if x > 1.0:
            x /= 100.0
        if conf is not None and conf > 1.0:
            conf /= 100.0
        if acc is not None and acc > 1.0:
            acc /= 100.0

        points.append(
            {
                "x": x,
                "confidence": conf if conf is not None else x,
                "accuracy": acc if acc is not None else 0.0,
                "count": count if count is not None else 1.0,
            }
        )
    return sorted(points, key=lambda point: point["x"])


def compute_ece(points: list[dict[str, float]]) -> float:
    if not points:
        return float("nan")
    total = sum(max(0.0, p["count"]) for p in points)
    if total <= 0:
        total = float(len(points))
    return sum((max(0.0, p["count"]) / total) * abs(p["accuracy"] - p["confidence"]) for p in points) * 100.0


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def plot_accuracy(ax, rows: list[dict[str, str]]) -> bool:
    series = build_accuracy_series(rows)
    if series is None:
        ax.text(0.5, 0.5, "No accuracy data found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    datasets, models, grouped = series
    x = list(range(len(datasets)))
    for model in models:
        y = []
        for dataset in datasets:
            values = grouped[dataset].get(model, [])
            y.append(sum(values) / len(values) if values else float("nan"))
        ax.plot(x, y, marker="o", linewidth=2.2, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cross-Dataset Accuracy")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    return True


def aggregate_slope_data(
    rows: list[dict[str, str]],
    metric: str,
) -> list[tuple[str, float, float]] | None:
    if not is_metric_table(rows) or metric not in rows[0]:
        return None

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"before": [], "after": []})
    for row in rows:
        before = scale_metric_value(to_float(row.get(f"{metric}_before")), metric)
        after = scale_metric_value(to_float(row.get(f"{metric}_after")), metric)
        if before is None and after is None:
            continue
        model = str(row.get("model", "")).strip() or "Model"
        if before is not None:
            grouped[model]["before"].append(before)
        if after is not None:
            grouped[model]["after"].append(after)

    if not grouped:
        return None

    series: list[tuple[str, float, float]] = []
    for model in sorted(grouped):
        before_vals = grouped[model]["before"]
        after_vals = grouped[model]["after"]
        before_avg = sum(before_vals) / len(before_vals) if before_vals else float("nan")
        after_avg = sum(after_vals) / len(after_vals) if after_vals else float("nan")
        series.append((model, before_avg, after_avg))
    return series


def plot_before_after_metric(ax, rows: list[dict[str, str]], metric: str, title: str, ylabel: str) -> bool:
    series = aggregate_slope_data(rows, metric)
    if series is None:
        ax.text(0.5, 0.5, f"No {metric} data found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    x = [0, 1]
    ax.set_xticks(x)
    ax.set_xticklabels(["Before", "After"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)

    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
    all_values = []
    for idx, (model, before, after) in enumerate(series):
        color = palette[idx % len(palette)]
        ax.plot(x, [before, after], marker="o", linewidth=2.2, color=color)
        if before == before:
            all_values.append(before)
        if after == after:
            all_values.append(after)
        if after == after:
            ax.text(1.05, after, f"{model} ({format_metric_value(after, metric)})", va="center", fontsize=8, color=color)

    if all_values:
        ymin = min(all_values)
        ymax = max(all_values)
        pad = (ymax - ymin) * 0.15 if ymax > ymin else (0.5 if metric == "ece" else 0.001)
        ax.set_ylim(max(0, ymin - pad), ymax + pad)

    ax.set_xlim(-0.05, 1.32)
    return True


def choose_reliability_pair(paths: list[Path]) -> tuple[Path | None, Path | None]:
    before = None
    after = None
    for path in paths:
        condition = reliability_condition_from_name(path.name)
        if condition == "before" and before is None:
            before = path
        elif condition == "after" and after is None:
            after = path

    if before is None and paths:
        before = paths[0]
    if after is None and len(paths) > 1:
        after = next((path for path in paths if path != before), None)
    return before, after


def plot_reliability_panel(ax, reliability_paths: list[Path]) -> bool:
    before_path, after_path = choose_reliability_pair(reliability_paths)
    if before_path is None and after_path is None:
        ax.text(0.5, 0.5, "No reliability data found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    if before_path is None:
        before_path = after_path
    if after_path is None:
        after_path = before_path

    before_rows = parse_reliability_rows(read_rows(before_path)) if before_path else []
    after_rows = parse_reliability_rows(read_rows(after_path)) if after_path else []

    def row_values(points: list[dict[str, float]]) -> tuple[list[float], list[float]]:
        xs = [p["x"] * 100.0 for p in points]
        ys = [p["accuracy"] * 100.0 for p in points]
        return xs, ys

    before_x, before_y = row_values(before_rows)
    after_x, after_y = row_values(after_rows)

    ax.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    if before_rows:
        ax.plot(before_x, before_y, marker="o", linewidth=2.2, color="#F58518", label=f"Before ({before_path.stem})")
    if after_rows:
        ax.plot(after_x, after_y, marker="s", linewidth=2.2, color="#54A24B", label=f"After ({after_path.stem})")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Reliability Diagram")
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Observed Accuracy (%)")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    return True


def collect_metric_series(rows: list[dict[str, str]], column: str, *, percent: bool = False) -> list[tuple[str, float]] | None:
    if not is_metric_table(rows) or column not in rows[0]:
        return None

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = to_float(row.get(column))
        if value is None:
            continue
        model = str(row.get("model", "")).strip() or "Model"
        if percent and value <= 1.0:
            value *= 100.0
        grouped[model].append(value)

    if not grouped:
        return None

    series = [(model, sum(values) / len(values)) for model, values in grouped.items()]
    series.sort(key=lambda item: item[1], reverse=True)
    return series


def plot_average_metric_bars(
    ax,
    rows: list[dict[str, str]],
    column: str,
    title: str,
    xlabel: str,
    *,
    percent: bool = False,
    value_fmt: str = ".2f",
    color: str = "#4C78A8",
) -> bool:
    series = collect_metric_series(rows, column, percent=percent)
    if series is None:
        ax.text(0.5, 0.5, f"No {column} data found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    models = [model for model, _ in series]
    values = [value for _, value in series]
    bars = ax.barh(models, values, color=color, alpha=0.9)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.2)

    max_value = max(values) if values else 0.0
    pad = max_value * 0.05 if max_value else 0.05
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + pad, bar.get_y() + bar.get_height() / 2, format(value, value_fmt), va="center", fontsize=8)
    ax.set_xlim(0, max_value + pad * 6)
    return True


def plot_calibration_improvement(
    ax,
    rows: list[dict[str, str]],
    metric: str = "ece",
    title: str = "Calibration Improvement (Before - After)",
) -> bool:
    if not is_metric_table(rows):
        ax.text(0.5, 0.5, "No calibration data found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    before_col = f"{metric}_before"
    after_col = f"{metric}_after"
    if before_col not in rows[0] or after_col not in rows[0]:
        ax.text(0.5, 0.5, f"No {metric} improvement data found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        before = scale_metric_value(to_float(row.get(before_col)), metric)
        after = scale_metric_value(to_float(row.get(after_col)), metric)
        if before is None or after is None:
            continue
        model = str(row.get("model", "")).strip() or "Model"
        grouped[model].append(before - after)

    if not grouped:
        ax.text(0.5, 0.5, f"No {metric} improvement data found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    series = [(model, sum(values) / len(values)) for model, values in grouped.items()]
    series.sort(key=lambda item: item[1], reverse=True)

    models = [model for model, _ in series]
    values = [value for _, value in series]
    bars = ax.barh(models, values, color="#54A24B", alpha=0.9)
    ax.axvline(0, color="gray", linewidth=1)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("ECE Reduction (Before - After, percentage points)")
    ax.grid(True, axis="x", alpha=0.2)

    max_abs = max(abs(v) for v in values) if values else 0.0
    pad = max_abs * 0.08 if max_abs else 0.05
    for bar, value in zip(bars, values):
        x = bar.get_width()
        offset = pad if value >= 0 else -pad
        ha = "left" if value >= 0 else "right"
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", ha=ha, fontsize=8)

    left = min(0.0, min(values) - pad)
    right = max(0.0, max(values) + pad * 4)
    ax.set_xlim(left, right)
    return True


def save_standalone_accuracy_figure(rows: list[dict[str, str]], output_dir: Path) -> Path | None:
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    ok = plot_accuracy(ax, rows)
    if not ok:
        plt.close(fig)
        return None

    out_path = output_dir / "cross_dataset_accuracy.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def save_average_metric_figure(
    rows: list[dict[str, str]],
    column: str,
    title: str,
    xlabel: str,
    filename: str,
    output_dir: Path,
    *,
    percent: bool = False,
    value_fmt: str = ".2f",
    color: str = "#4C78A8",
) -> Path | None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ok = plot_average_metric_bars(ax, rows, column, title, xlabel, percent=percent, value_fmt=value_fmt, color=color)
    if not ok:
        plt.close(fig)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def save_calibration_improvement_figure(
    rows: list[dict[str, str]],
    filename: str,
    output_dir: Path,
    *,
    metric: str = "ece",
) -> Path | None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    title = f"Calibration Improvement in {metric.upper()} (Before - After)"
    ok = plot_calibration_improvement(ax, rows, metric=metric, title=title)
    if not ok:
        plt.close(fig)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def save_standalone_metric_figure(
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ok = plot_before_after_metric(ax, rows, metric, title, ylabel)
    if not ok:
        plt.close(fig)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def save_reliability_figure(reliability_csvs: list[Path], output_dir: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ok = plot_reliability_panel(ax, reliability_csvs)
    if not ok:
        plt.close(fig)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "reliability_before_after.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def build_poster_panel(results_csv: Path, reliability_csvs: list[Path], output_dir: Path) -> Path | None:
    rows = read_rows(results_csv)
    if not rows:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    save_standalone_accuracy_figure(rows, output_dir)
    save_average_metric_figure(
        rows,
        "accuracy",
        "Average Accuracy by Model",
        "Accuracy (%)",
        "average_accuracy_by_model.png",
        output_dir,
        percent=True,
        value_fmt=".1f",
        color="#4C78A8",
    )
    save_average_metric_figure(
        rows,
        "ece_after",
        "Average ECE by Model (After Temperature Scaling)",
        "ECE (%)",
        "average_ece_by_model.png",
        output_dir,
        percent=True,
        value_fmt=".2f",
        color="#F58518",
    )
    save_average_metric_figure(
        rows,
        "brier_after",
        "Average Brier Score by Model (After Temperature Scaling)",
        "Brier Score",
        "average_brier_by_model.png",
        output_dir,
        percent=False,
        value_fmt=".4f",
        color="#B279A2",
    )
    save_calibration_improvement_figure(rows, "calibration_improvement_ece.png", output_dir, metric="ece")
    save_standalone_metric_figure(
        rows,
        "ece",
        "ECE Before vs After",
        "ECE (%)",
        "ece_before_after.png",
        output_dir,
    )
    save_standalone_metric_figure(
        rows,
        "brier",
        "Brier Score Before vs After",
        "Brier Score",
        "brier_before_after.png",
        output_dir,
    )
    save_reliability_figure(reliability_csvs, output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    plot_accuracy(axes[0], rows)
    plot_before_after_metric(axes[1], rows, "ece", "ECE Before vs After", "ECE (%)")
    plot_before_after_metric(axes[2], rows, "brier", "Brier Score Before vs After", "Brier Score")
    plot_reliability_panel(axes[3], reliability_csvs)

    fig.suptitle("Poster Results Summary", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = output_dir / "poster_results_summary.png"
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate poster-ready calibration figures.")
    parser.add_argument("--results-csv", type=Path, default=None, help="Path to master_results_table.csv")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory containing master_results_table.csv")
    parser.add_argument("--reliability-csv", type=Path, nargs="*", default=None, help="One or more reliability CSV files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated figures")
    args = parser.parse_args()

    results_csv = resolve_results_csv(args.results_csv, args.results_dir)

    if results_csv is None:
        print("No figures were generated. Could not find master_results_table.csv.")
        return

    rows = read_rows(results_csv)
    reliability_csvs = resolve_reliability_csvs(args.reliability_csv, args.results_dir, rows)

    out_path = build_poster_panel(results_csv, reliability_csvs, args.output_dir)
    if out_path is None:
        print("No figures were generated. Check that the results and reliability CSV files exist.")
    else:
        print("Generated poster figures:")
        print(f" - {args.output_dir / 'cross_dataset_accuracy.png'}")
        print(f" - {args.output_dir / 'average_accuracy_by_model.png'}")
        print(f" - {args.output_dir / 'average_ece_by_model.png'}")
        print(f" - {args.output_dir / 'average_brier_by_model.png'}")
        print(f" - {args.output_dir / 'calibration_improvement_ece.png'}")
        print(f" - {args.output_dir / 'ece_before_after.png'}")
        print(f" - {args.output_dir / 'brier_before_after.png'}")
        print(f" - {args.output_dir / 'reliability_before_after.png'}")
        print(f" - {out_path}")


if __name__ == "__main__":
    main()
