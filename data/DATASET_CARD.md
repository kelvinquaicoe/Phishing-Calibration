# Dataset Card: EduPhish and Related Phishing Corpora

## Summary
This repository contains phishing and legitimate email datasets prepared for research, calibration, and classification experiments.

The main curated dataset is **EduPhish**, an education-targeted phishing dataset derived from publicly available phishing corpora and filtered for education-context content.

## Repository contents
- `Education-Targeted Phishing Email Dataset/eduphish_dataset.csv`
- `train_eduphish.csv`
- `test_eduphish.csv`
- `distilbert_eduphish_train.csv`
- `distilbert_eduphish_validation.csv`
- `train_phishing_email.csv`
- `test_phishing_email.csv`
- `train_phishing_legit.csv`
- `test_phishing_legit.csv`
- `phishing_legit_dataset_KD_10000.csv`
- `graphs/` summary figures

## Main dataset: EduPhish
### Size
- Total samples: 16,942
- Phishing emails: 7,817
- Legitimate emails: 9,125

### Schema
- `text`: email content
- `label`: binary class label where `0 = legitimate` and `1 = phishing`

### Construction steps
1. Aggregated multiple public phishing corpora.
2. Combined subject and body text where available.
3. Filtered for education-related content.
4. Applied authentication / credential-related keyword filtering.
5. Removed duplicates using normalized hashing.
6. Performed stratified sampling to balance classes.

## Knowledge-distilled dataset
### File
- `phishing_legit_dataset_KD_10000.csv`

### Size
- 10,000 samples

### Schema
- `text`
- `label`
- `phishing_type`
- `severity`
- `confidence`

### Label distribution
- Phishing types: 11 categories
- Severity values: low / medium / high
- Average confidence: ~0.899

## Intended use
Recommended for:
- phishing detection research
- binary and multi-class classification experiments
- calibration studies
- NLP benchmarking
- cybersecurity education and experimentation

## Not intended for
- malicious use
- phishing campaign development
- credential harvesting
- spam distribution

## Data sources
This repository aggregates or derives content from publicly available corpora, including:
- Hugging Face: `puyang2025/seven-phishing-email-datasets`
- Hugging Face: `amrithanandini/phishing-email-rich-dataset-v2`
- Hugging Face: `zefang-liu/phishing-email-dataset`
- Kaggle: `subhajournal/phishingemails`

Each upstream source retains its own license and notice requirements.

## Licensing note
This repository is a derivative compilation. Users must comply with the upstream licenses and preserve original notices.

See:
- `Education-Targeted Phishing Email Dataset/LICENSE_NOTICE.txt`
- `Education-Targeted Phishing Email Dataset/README.md`

## Generated figures
Run:

```bash
python3 generate_graphs.py
```

to regenerate the charts in `graphs/`.

## Suggested citation
If you use this repository in a project or paper, please cite the repository and the upstream data sources used to create it.
