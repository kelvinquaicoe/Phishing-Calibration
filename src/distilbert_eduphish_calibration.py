from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "results" / "distilbert_eduphish" / "checkpoint-1695"
VALIDATION_FILE = PROJECT_ROOT / "data" / "distilbert_eduphish_validation.csv"
TEST_FILE = PROJECT_ROOT / "data" / "test_eduphish.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "distilbert_calibration_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_OUTPUT = OUTPUT_DIR / "distilbert_validation_predictions.csv"
TEST_OUTPUT = OUTPUT_DIR / "distilbert_test_predictions.csv"
TEST_SCALED_OUTPUT = OUTPUT_DIR / "distilbert_test_predictions_scaled.csv"

MAX_LENGTH = 256
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        item["text"] = text
        return item


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        before = nll(logits, labels).item()

        def closure():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        after = nll(self.forward(logits), labels).item()
        temp = float(self.temperature.item())

        print(f"Validation NLL before scaling: {before:.4f}")
        print(f"Validation NLL after scaling:  {after:.4f}")
        print(f"Learned temperature: {temp:.4f}")

        return temp


def _raise_csv_field_limit() -> None:
    # Some datasets include very long email bodies; the default csv module limit (131072)
    # can be too small. Increase it safely across platforms.
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size = int(max_size / 10)


def read_csv_file(path: Path) -> Tuple[List[str], List[int]]:
    _raise_csv_field_limit()
    texts, labels = [], []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def load_tokenizer(model_dir: Path):
    # Prefer a locally available tokenizer first to avoid unnecessary network calls in
    # restricted environments, then fall back to downloading the base tokenizer.
    candidates = [str(model_dir), "distilbert-base-uncased"]
    last_error: Exception | None = None
    for candidate in candidates:
        for local_only in (True, False):
            try:
                return AutoTokenizer.from_pretrained(candidate, local_files_only=local_only)
            except Exception as exc:
                last_error = exc

    raise RuntimeError(
        "Unable to load tokenizer. If you're running offline, pre-download "
        "'distilbert-base-uncased' (or add tokenizer files to the checkpoint directory)."
    ) from last_error


def infer(model, tokenizer, texts: List[str], labels: List[int]):
    dataset = TextDataset(texts, labels, tokenizer, max_length=MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    softmax = nn.Softmax(dim=-1)

    all_logits = []
    all_labels = []
    rows = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            y = batch["labels"].to(DEVICE)
            texts_batch = batch["text"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = softmax(logits)
            pred = probs.argmax(dim=-1)
            conf = probs.max(dim=-1).values

            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

            for i, text in enumerate(texts_batch):
                rows.append([
                    text,
                    int(y[i].item()),
                    int(pred[i].item()),
                    f"{float(conf[i].item()):.6f}",
                    f"{float(probs[i][0].item()):.6f}",
                    f"{float(probs[i][1].item()):.6f}",
                    f"{float(logits[i][0].item()):.6f}",
                    f"{float(logits[i][1].item()):.6f}",
                ])

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0), rows


def scaled_infer(model, tokenizer, texts: List[str], labels: List[int], temperature: float):
    dataset = TextDataset(texts, labels, tokenizer, max_length=MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    softmax = nn.Softmax(dim=-1)
    rows = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            y = batch["labels"].to(DEVICE)
            texts_batch = batch["text"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            scaled_logits = logits / temperature
            probs = softmax(scaled_logits)
            pred = probs.argmax(dim=-1)
            conf = probs.max(dim=-1).values

            for i, text in enumerate(texts_batch):
                rows.append([
                    text,
                    int(y[i].item()),
                    int(pred[i].item()),
                    f"{float(conf[i].item()):.6f}",
                    f"{float(probs[i][0].item()):.6f}",
                    f"{float(probs[i][1].item()):.6f}",
                    f"{float(scaled_logits[i][0].item()):.6f}",
                    f"{float(scaled_logits[i][1].item()):.6f}",
                ])

    return rows


def write_rows(path: Path, header: List[str], rows: List[List[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_DIR}")
    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(f"Validation file not found: {VALIDATION_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    print(f"Using model: {MODEL_DIR}")
    print(f"Using device: {DEVICE}")
    print(f"Validation file: {VALIDATION_FILE}")
    print(f"Test file: {TEST_FILE}")

    tokenizer = load_tokenizer(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

    val_texts, val_labels = read_csv_file(VALIDATION_FILE)
    test_texts, test_labels = read_csv_file(TEST_FILE)

    val_logits, val_y, val_rows = infer(model, tokenizer, val_texts, val_labels)
    write_rows(
        VALIDATION_OUTPUT,
        ["text", "label", "pred_label", "confidence", "prob_0", "prob_1", "logit_0", "logit_1"],
        val_rows,
    )
    print(f"Saved validation predictions -> {VALIDATION_OUTPUT}")

    scaler = TemperatureScaler().to(DEVICE)
    temperature = scaler.fit(val_logits.to(DEVICE), val_y.to(DEVICE))

    _, _, test_rows = infer(model, tokenizer, test_texts, test_labels)
    write_rows(
        TEST_OUTPUT,
        ["text", "label", "pred_label", "confidence", "prob_0", "prob_1", "logit_0", "logit_1"],
        test_rows,
    )
    print(f"Saved test predictions -> {TEST_OUTPUT}")

    scaled_rows = scaled_infer(model, tokenizer, test_texts, test_labels, temperature)
    write_rows(
        TEST_SCALED_OUTPUT,
        ["text", "label", "pred_label", "confidence", "prob_0", "prob_1", "scaled_logit_0", "scaled_logit_1"],
        scaled_rows,
    )
    print(f"Saved scaled test predictions -> {TEST_SCALED_OUTPUT}")

    print("Done. Next step: compute accuracy and calibration metrics from the saved CSV files.")


if __name__ == "__main__":
    main()
