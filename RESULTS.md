# Results

This file summarizes the **calibration** results for the EduPhish test set, before and after **temperature scaling**.

## Figures

Dataset exploration figures are stored in `graphs/datasets/`.

Research result figures are stored in `graphs/results/`.

Key result figures include:

- `cross_dataset_accuracy.png` — compares model accuracy across EduPhish, CEAS_08, Enron, and SpamAssassin.
- `average_accuracy_by_model.png` — summarizes average model accuracy across all four datasets.
- `average_ece_by_model.png` — summarizes average Expected Calibration Error after temperature scaling.
- `average_brier_by_model.png` — summarizes average Brier Score after temperature scaling.
- `calibration_improvement_ece.png` — shows how much ECE decreased after temperature scaling.
- `reliability_before_after.png` — shows calibration before and after temperature scaling using standard confidence bins.
- `poster_results_summary.png` — combined results panel for poster use.

## EduPhish calibration (test set)

| Model | Accuracy | ECE (before) | ECE (after) | Brier (before) | Brier (after) |
|---|---:|---:|---:|---:|---:|
| BERT | 0.9680 | 0.0223 | 0.0205 | 0.0261 | 0.0256 |
| DistilBERT | 0.9930 | 0.0058 | 0.0052 | 0.0063 | 0.0063 |
| RoBERTa | 0.9650 | 0.0314 | 0.0302 | 0.0319 | 0.0311 |
| ALBERT | 0.6770 | 0.2359 | 0.2125 | 0.2747 | 0.2626 |
| DeBERTa-v3-small | 0.4640 | 0.5283 | 0.5145 | 0.5268 | 0.5122 |
| DeBERTa-v3-base | 0.4630 | 0.5241 | 0.5079 | 0.5246 | 0.5084 |


## Overview

A total of six lightweight phishing email detectors were evaluated across four datasets: EduPhish, CEAS_08, Enron, and SpamAssassin. For each model–dataset pair, calibration was measured before and after temperature scaling using Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Negative Log-Likelihood (NLL), Brier Score, and reliability diagrams.

Across all experiments, temperature scaling consistently improved the quality of the predicted probabilities. While classification accuracy remained unchanged, the calibrated models generally produced lower NLL, lower ECE, and lower Brier scores, indicating that the confidence estimates more accurately reflected the true likelihood of correct predictions.

## EduPhish

EduPhish served as the in-distribution evaluation dataset. DistilBERT achieved the highest classification accuracy (99.3%) while also producing the lowest Expected Calibration Error and Brier Score, making it the best-calibrated model on the training distribution. BERT and RoBERTa also demonstrated strong performance, with temperature scaling providing modest improvements in calibration.

ALBERT produced substantially lower classification accuracy than the other transformer models, although its calibration metrics improved after temperature scaling. The two DeBERTa checkpoints performed poorly on EduPhish, suggesting that these checkpoints were not well matched to this dataset or require further investigation.

## CEAS_08

Performance changed noticeably on the CEAS_08 dataset. DeBERTa-v3-base achieved the highest classification accuracy (88.76%), followed by DeBERTa-v3-small (85.12%) and RoBERTa (83.46%). This contrasts with the EduPhish results and demonstrates that model ranking depends strongly on the evaluation corpus.

Temperature scaling reduced calibration error for nearly every model while leaving classification accuracy unchanged.

## Enron

The Enron dataset produced a different ranking of models. BERT achieved the highest accuracy (95.12%), followed closely by RoBERTa, DistilBERT, and DeBERTa-v3-small. ALBERT again exhibited noticeably lower classification performance.

These results demonstrate that no single architecture consistently outperformed all others across every dataset.

## SpamAssassin

On SpamAssassin, DeBERTa-v3-small achieved the highest accuracy (94.00%), narrowly outperforming DistilBERT and BERT. RoBERTa remained competitive, while ALBERT and DeBERTa-v3-base showed lower performance than the remaining transformer models.

As with the other datasets, temperature scaling generally reduced calibration error and improved probability estimates without changing classification accuracy.

## Discussion

The result figures show that no single model dominates across all datasets. BERT, DistilBERT, and RoBERTa achieve the strongest average accuracy, while calibration metrics such as ECE and Brier Score reveal additional differences in probability quality. Temperature scaling generally lowers calibration error without changing classification accuracy.

The experiment demonstrates that confidence calibration is highly dependent on dataset characteristics. A model that performs best on one corpus is not necessarily the strongest performer on another. This variability highlights the importance of evaluating phishing detectors across multiple datasets rather than relying on a single benchmark.

Temperature scaling proved to be an effective calibration method. Across nearly all experiments, it reduced Expected Calibration Error, Negative Log-Likelihood, and Brier Score while preserving the underlying classifier's accuracy. These findings suggest that inexpensive calibration techniques can substantially improve the reliability of confidence scores for lightweight phishing email detectors deployed in practice.


