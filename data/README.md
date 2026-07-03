# Phishing Calibration Dataset Repository

This repository contains curated phishing-related datasets, train/test splits, and summary graphs for calibration and classification experiments.

It is organized so you can upload the raw data, the derived splits, and the figures directly to GitHub.

## Quick facts

- Main dataset: EduPhish
- Total samples: 16,942
- Binary labels: legitimate (0) / phishing (1)
- Extra KD dataset: 10,000 samples with phishing type, severity, and confidence
- Summary charts: saved in `graphs/`

## What is included

### 1) EduPhish dataset
Location:
- `Education-Targeted Phishing Email Dataset/eduphish_dataset.csv`

Contents:
- education-targeted phishing emails
- legitimate emails
- cleaned and deduplicated text samples

### 2) Dataset splits
- `train_eduphish.csv` — 4,000 rows
- `test_eduphish.csv` — 1,000 rows
- `distilbert_eduphish_train.csv` — 3,200 rows
- `distilbert_eduphish_validation.csv` — 800 rows
- `train_phishing_email.csv` — 4,000 rows
- `test_phishing_email.csv` — 1,000 rows
- `train_phishing_legit.csv` — 4,000 rows
- `test_phishing_legit.csv` — 1,000 rows

### 3) Knowledge-distilled phishing dataset
- `phishing_legit_dataset_KD_10000.csv` — 10,000 rows

Includes:
- `text`
- `label`
- `phishing_type`
- `severity`
- `confidence`

Summary:
- phishing types: 11 classes
- severity labels: low / medium / high
- average confidence: ~0.899

### 4) Reference source corpus
- `Phishing Email Dataset/`

This folder contains the original source corpora used for experimentation and comparison.

### 5) Graphs
Generated graphs are saved in:
- `graphs/`

### 6) Dataset card, citation, and license
- `DATASET_CARD.md`
- `CITATION.cff`
- `LICENSE.md`

These files help make the GitHub repository easier to understand, cite, and publish.

Current graphs:
- `dataset_sizes.png`
- `eduphish_class_distribution.png`
- `eduphish_train_class_distribution.png`
- `eduphish_test_class_distribution.png`
- `kd_phishing_type_distribution.png`
- `kd_severity_distribution.png`
- `kd_confidence_distribution.png`
- `kd_text_length_by_class.png`

These cover:
- dataset size comparison
- class distribution charts
- phishing type distribution
- severity distribution
- confidence distribution
- text-length distribution by class

## How to generate the graphs

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python3 generate_graphs.py
```

This will create PNG files in the `graphs/` folder.

## Recommended GitHub structure

```text
.
├── Education-Targeted Phishing Email Dataset/
├── Phishing Email Dataset/
├── graphs/
├── README.md
├── generate_graphs.py
├── train_eduphish.csv
├── test_eduphish.csv
├── distilbert_eduphish_train.csv
├── distilbert_eduphish_validation.csv
├── train_phishing_email.csv
├── test_phishing_email.csv
├── train_phishing_legit.csv
├── test_phishing_legit.csv
└── phishing_legit_dataset_KD_10000.csv
```

## Notes

- The dataset files are intended for research and calibration experiments.
- Keep upstream license notices with the source datasets.
- If you publish this on GitHub, add a short license note in the repository root describing the upstream dataset terms.
- The repo includes a `requirements.txt` file for graph generation.
- The `DATASET_CARD.md` file is a good place to summarize the dataset for GitHub viewers.
- The `CITATION.cff` file helps GitHub show citation metadata.
- The `LICENSE.md` file provides a repository-level licensing note.
