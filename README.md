# Phishing-Calibration

Lightweight transformer phishing-email detectors with a focus on **confidence calibration** (temperature scaling) and reliability diagrams.

## Repository layout

- `data/` — datasets, train/test splits, graphs, dataset card + citation (tracked; CSVs use Git LFS)
- `src/` — training, calibration, metrics, and plotting scripts (tracked)
- `docs/` — short writeups (models + datasets)
- `results/` — **local-only outputs** (ignored; often large checkpoints + prediction files)
- `models/` — **local-only cached model/tokenizer files** (ignored)

## Setup

1) Install Git LFS and pull large CSVs:

```bash
git lfs install
git lfs pull
```

2) Create a virtualenv and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers datasets scikit-learn pandas matplotlib
```

## Run: calibration → metrics → plots (EduPhish)

Each model has three scripts:

- calibration: writes prediction CSVs (before/after temperature scaling)
- metrics: writes JSON metrics + reliability CSV
- plots: writes reliability diagram + confidence histogram PNGs

Examples:

```bash
python3 src/distilbert_eduphish_calibration.py
python3 src/distilbert_eduphish_metrics.py
python3 src/distilbert_eduphish_plots.py
```

```bash
python3 src/roberta_eduphish_calibration.py
python3 src/roberta_eduphish_metrics.py
python3 src/roberta_eduphish_plots.py
```

Outputs land under:

- `results/distilbert_calibration_outputs/`
- `results/roberta_calibration_outputs/`
- `results/bert_calibration_outputs/`
- `results/albert_calibration_outputs/`
- `results/deberta_calibration_outputs/`
- `results/deberta_v3_base_calibration_outputs/`

## Notes

- See `docs/models.md` for the model panel and `docs/datasets.md` for dataset files.
- See `RESULTS.md` for the latest summarized calibration metrics.

