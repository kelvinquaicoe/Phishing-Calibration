# Datasets

All dataset files live under `data/` (and are versioned; `*.csv` files are stored with Git LFS).

## EduPhish (education-targeted phishing)

- Raw dataset: `data/Education-Targeted Phishing Email Dataset/eduphish_dataset.csv`
- Splits used in experiments:
  - `data/train_eduphish.csv`
  - `data/test_eduphish.csv`
  - `data/distilbert_eduphish_train.csv`
  - `data/distilbert_eduphish_validation.csv`

The calibration scripts typically use the validation split to fit temperature scaling and the test split for final evaluation.

## Phishing Email Dataset (source corpora)

Folder: `data/Phishing Email Dataset/`

Included source corpora CSVs:

- `data/Phishing Email Dataset/CEAS_08.csv`
- `data/Phishing Email Dataset/Enron.csv`
- `data/Phishing Email Dataset/Ling.csv`
- `data/Phishing Email Dataset/Nazario.csv`
- `data/Phishing Email Dataset/Nigerian_Fraud.csv`
- `data/Phishing Email Dataset/SpamAssasin.csv`
- `data/Phishing Email Dataset/phishing_email.csv`

## Additional labeled splits (binary)

- `data/train_phishing_email.csv` / `data/test_phishing_email.csv`
- `data/train_phishing_legit.csv` / `data/test_phishing_legit.csv`

## Knowledge-distilled (KD) dataset

- `data/phishing_legit_dataset_KD_10000.csv`

Fields include `text`, `label`, and additional metadata (e.g., phishing type, severity, confidence).

## Dataset documentation

- Dataset card: `data/DATASET_CARD.md`
- Manifest: `data/DATA_MANIFEST.md`
- Citation: `data/CITATION.cff`
- License: `data/LICENSE.md`
- Dataset graphs: `data/graphs/`
