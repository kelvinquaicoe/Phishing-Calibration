# Models Evaluated

This project evaluates the confidence calibration of six lightweight transformer-based models for phishing email detection. These models were selected because they represent a range of transformer architectures with different parameter counts, training strategies, and computational requirements while remaining practical for deployment on commodity hardware.

---

## BERT

**Architecture:** Bidirectional Encoder Representations from Transformers (BERT)

BERT is one of the foundational transformer architectures for natural language processing. It uses bidirectional self-attention to learn contextual representations of text and serves as a strong baseline for text classification tasks.

**Why included**

* Widely used benchmark model
* Strong performance on text classification
* Serves as a reference point for comparison with newer architectures

---

## DistilBERT

**Architecture:** Distilled version of BERT

DistilBERT is a compressed version of BERT created using knowledge distillation. It retains much of BERT's performance while requiring fewer parameters and less computation.

**Why included**

* Lightweight and efficient
* Faster inference than BERT
* Suitable for deployment on resource-constrained systems

---

## RoBERTa

**Architecture:** Robustly Optimized BERT Approach

RoBERTa improves upon the original BERT training procedure by using larger training datasets, dynamic masking, and optimized hyperparameters.

**Why included**

* Strong performance across many NLP tasks
* Common benchmark for transformer-based classifiers
* Provides comparison against the original BERT architecture

---

## ALBERT

**Architecture:** A Lite BERT

ALBERT reduces model size by sharing parameters across transformer layers and factorizing the embedding matrix. This significantly lowers memory requirements while maintaining competitive performance.

**Why included**

* Reduced memory footprint
* Efficient architecture
* Represents parameter-sharing transformer designs

---

## DeBERTa-v3-small

**Architecture:** Decoding-enhanced BERT with Disentangled Attention (Small)

DeBERTa-v3 introduces disentangled attention, allowing the model to separately represent word content and positional information. The small version is designed for efficient deployment while leveraging these architectural improvements.

**Why included**

* Modern transformer architecture
* Lightweight variant
* Evaluates whether newer attention mechanisms improve confidence calibration

---

## DeBERTa-v3-base

**Architecture:** Decoding-enhanced BERT with Disentangled Attention (Base)

The base version of DeBERTa-v3 contains more parameters than the small variant, providing higher model capacity while maintaining the same underlying architecture.

**Why included**

* Higher-capacity version of DeBERTa-v3
* Allows comparison between small and base model sizes
* Evaluates whether increased model capacity affects calibration

---

# Model Comparison

| Model            | Relative Size | Primary Goal                    |
| ---------------- | ------------- | ------------------------------- |
| BERT             | Medium        | Baseline transformer            |
| DistilBERT       | Small         | Efficient inference             |
| RoBERTa          | Medium        | Improved BERT training          |
| ALBERT           | Small         | Parameter-efficient transformer |
| DeBERTa-v3-small | Small         | Modern lightweight architecture |
| DeBERTa-v3-base  | Medium        | Higher-capacity DeBERTa model   |

---

The objective of this project is not to determine which model achieves the highest classification accuracy alone. Instead, the goal is to compare how well each model's confidence scores correspond to its true prediction accuracy.

For every model, the following calibration metrics are evaluated before and after temperature scaling:

* Accuracy
* Expected Calibration Error (ECE)
* Maximum Calibration Error (MCE)
* Brier Score
* Negative Log-Likelihood (NLL)
* Reliability Diagrams


