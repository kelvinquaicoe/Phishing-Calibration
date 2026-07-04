# Results

This file summarizes the **calibration** results for the EduPhish test set, before and after **temperature scaling**.

## EduPhish calibration (test set)

| Model | Accuracy | ECE (before) | ECE (after) | Brier (before) | Brier (after) |
|---|---:|---:|---:|---:|---:|
| BERT | 0.9680 | 0.0223 | 0.0205 | 0.0261 | 0.0256 |
| DistilBERT | 0.9930 | 0.0058 | 0.0052 | 0.0063 | 0.0063 |
| RoBERTa | 0.9650 | 0.0314 | 0.0302 | 0.0319 | 0.0311 |
| ALBERT | 0.6770 | 0.2359 | 0.2125 | 0.2747 | 0.2626 |
| DeBERTa-v3-small | 0.4640 | 0.5283 | 0.5145 | 0.5268 | 0.5122 |
| DeBERTa-v3-base | 0.4630 | 0.5241 | 0.5079 | 0.5246 | 0.5084 |

Notes:

- Temperature scaling consistently improves NLL/ECE, but it cannot fix a model that is systematically biased (e.g., predicting almost everything as phishing).
- The DeBERTa checkpoints used here behave pathologically on this split (near-all-positive predictions), so these entries likely need checkpoint debugging/retraining to be meaningful.

