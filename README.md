# Confidence Calibration in Lightweight Phishing Email Detectors: A Cross-Dataset Analysis

Lightweight transformer-based phishing email detectors with a focus on **confidence calibration**, **temperature scaling**, and **cross-dataset evaluation**.

This project investigates whether modern phishing email classifiers produce confidence scores that accurately reflect their true prediction accuracy. While transformer models often achieve high classification accuracy, their confidence estimates may become unreliable when evaluated on different phishing datasets. This repository provides a reproducible evaluation framework for measuring calibration before and after temperature scaling.

---

# Research Objective

This project evaluates six lightweight transformer architectures across multiple phishing email datasets to answer the following question:

> **When a phishing detector predicts an email is phishing with high confidence, can that confidence score actually be trusted across different datasets?**

Calibration is evaluated before and after temperature scaling using:

* Accuracy
* Negative Log-Likelihood (NLL)
* Expected Calibration Error (ECE)
* Maximum Calibration Error (MCE)
* Brier Score
* Reliability Diagrams
* Confidence Histograms

---

# Models Evaluated

* BERT
* DistilBERT
* RoBERTa
* ALBERT
* DeBERTa-v3-small
* DeBERTa-v3-base

---

# Datasets

Experiments were performed using:

* EduPhish
* CEAS_08
* Enron
* SpamAssassin

Balanced evaluation subsets were created for the larger datasets to provide consistent comparisons while reducing evaluation time.

---

# Repository Layout

```text
data/      - Datasets, train/test splits, graphs, dataset cards, citations
docs/      - Project documentation (models, datasets, methodology)
models/    - Local Hugging Face model/tokenizer files (ignored by Git)
results/   - Calibration outputs, metrics, reliability diagrams, plots
src/       - Training, calibration, evaluation, and automation scripts
```

---

# Setup

## Clone the repository

```bash
git clone <repository-url>
cd phishing-calibration
```

## Install Git LFS

Large CSV datasets are tracked using Git LFS.

```bash
git lfs install
git lfs pull
```

## Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install dependencies

```bash
python -m pip install --upgrade pip

python -m pip install \
torch \
transformers \
datasets \
scikit-learn \
pandas \
matplotlib
```

---

# Running Experiments

## Run calibration

```bash
python3 src/run_calibration.py \
  --model-dir <checkpoint> \
  --validation-file <validation.csv> \
  --test-file <dataset.csv> \
  --output-dir <results_directory> \
  --prefix <model_name>
```

## Run all calibration experiments

```bash
python3 src/run_all_experiments.py
```

## Compute calibration metrics

```bash
python3 src/run_all_metrics.py
```

## Build the master results table

```bash
python3 src/build_master_results_table.py
```

---

# EduPhish Workflow

Each model follows the same three-stage pipeline:

1. Calibration
2. Metrics
3. Reliability plots

Example:

```bash
python3 src/distilbert_eduphish_calibration.py
python3 src/distilbert_eduphish_metrics.py
python3 src/distilbert_eduphish_plots.py
```

The same workflow applies to the remaining models.

---

# Outputs

Calibration outputs are written to:

```text
results/bert_calibration_outputs/
results/distilbert_calibration_outputs/
results/roberta_calibration_outputs/
results/albert_calibration_outputs/
results/deberta_calibration_outputs/
results/deberta_v3_base_calibration_outputs/
```

Additional cross-dataset experiments generate separate output directories for each model–dataset combination.

---

# Documentation

Additional documentation is available in:

* `docs/models.md`
* `docs/datasets.md`
* `RESULTS.md`

The complete experimental results are also available in:

* `results/master_results_table.csv`
* `results/master_results_table.md`

---

# Technologies

* Python
* PyTorch
* Hugging Face Transformers
* pandas
* scikit-learn
* Matplotlib

---

# Author

Kelvin Quaicoe

Summer Research Project

**Confidence Calibration in Lightweight Phishing Email Detectors: A Cross-Dataset Analysis**

