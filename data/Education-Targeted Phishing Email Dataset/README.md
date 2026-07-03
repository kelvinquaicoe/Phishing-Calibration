# EduPhish: Education-Targeted Phishing Email Dataset

## Overview
EduPhish is a curated phishing detection dataset focused on education-targeted attacks.
The dataset was constructed from multiple publicly available phishing corpora and filtered
to retain education-context phishing and legitimate emails.

## Data Sources
The dataset was derived from publicly available corpora including:

- Hugging Face: puyang2025/seven-phishing-email-datasets
- Hugging Face: amrithanandini/phishing-email-rich-dataset-v2
- Hugging Face: zefang-liu/phishing-email-dataset
- Kaggle: subhajournal/phishingemails

Each source retains its original license.

## Dataset Construction Methodology
1. Aggregated multiple public phishing corpora.
2. Combined subject and body text where available.
3. Applied education-related keyword filtering.
4. Applied authentication/credential-related keyword filtering.
5. Removed duplicates using normalized hashing.
6. Performed stratified sampling to balance classes.

## Final Statistics
- Phishing Emails: 7,817
- Safe Emails: 9,125
- Total: 16,942

## Licensing Notice

This dataset is a derivative compilation of publicly available datasets.

Redistribution is provided under the terms of the original dataset licenses.
Users must comply with all upstream licensing requirements.

The authors of EduPhish do not claim ownership of the original email content.

## Intended Use

This dataset is designed for:

Academic research

Phishing detection modeling

Natural Language Processing research

Cybersecurity experimentation

It is not intended for malicious use.

## Ethical Statement

EduPhish contains anonymized phishing and legitimate email samples from historical public corpora.
No private scraping or personal email harvesting was conducted.

Generated on: 2026-02-23