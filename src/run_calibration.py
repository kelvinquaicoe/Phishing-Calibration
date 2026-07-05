from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "results" / "roberta_eduphish" / "checkpoint-500"
VALIDATION_FILE = PROJECT_ROOT / "data" / "distilbert_eduphish_validation.csv"
TEST_FILE = PROJECT_ROOT / "data" / "test_eduphish.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "roberta_calibration_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

validation_output = OUTPUT_DIR / "roberta_validation_predictions.csv"
test_output = OUTPUT_DIR / "roberta_test_predictions.csv"
test_scaled_output = OUTPUT_DIR / "roberta_test_predictions_scaled.csv"

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

        print(f"Validation NLL before scaling: {before:.4f}", flush=True)
        print(f"Validation NLL after scaling:  {after:.4f}", flush=True)
        print(f"Learned temperature: {temp:.4f}", flush=True)

        return temp


def _raise_csv_field_limit() -> None:
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
            text_value = row.get("text") or row.get("body") or row.get("text_combined")
            if text_value is None:
                raise KeyError(f"Could not find text column in {path}. Expected one of: text, body, text_combined")
            texts.append(text_value)
            labels.append(int(row["label"]))
    return texts, labels


def load_tokenizer(model_dir: Path):
    candidates = [str(model_dir), "roberta-base"]
    last_error: Exception | None = None
    for candidate in candidates:
        for local_only in (True, False):
            try:
                return AutoTokenizer.from_pretrained(candidate, local_files_only=local_only)
            except Exception as exc:
                last_error = exc

    raise RuntimeError(
        "Unable to load tokenizer. If you're running offline, pre-download "
        "'roberta-base' (or add tokenizer files to the checkpoint directory)."
    ) from last_error


def infer(model, tokenizer, texts: List[str], labels: List[int], stage_name: str):
    dataset = TextDataset(texts, labels, tokenizer, max_length=MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    softmax = nn.Softmax(dim=-1)

    all_logits = []
    all_labels = []
    rows = []

    total_batches = len(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == total_batches:
                print(f"[{stage_name}] batch {batch_idx}/{total_batches}", flush=True)

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
                rows.append(
                    [
                        text,
                        int(y[i].item()),
                        int(pred[i].item()),
                        f"{float(conf[i].item()):.6f}",
                        f"{float(probs[i][0].item()):.6f}",
                        f"{float(probs[i][1].item()):.6f}",
                        f"{float(logits[i][0].item()):.6f}",
                        f"{float(logits[i][1].item()):.6f}",
                    ]
                )

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0), rows


def scaled_infer(model, tokenizer, texts: List[str], labels: List[int], temperature: float, stage_name: str):
    dataset = TextDataset(texts, labels, tokenizer, max_length=MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    softmax = nn.Softmax(dim=-1)

    rows = []

    total_batches = len(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == total_batches:
                print(f"[{stage_name}] batch {batch_idx}/{total_batches}", flush=True)

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
                rows.append(
                    [
                        text,
                        int(y[i].item()),
                        int(pred[i].item()),
                        f"{float(conf[i].item()):.6f}",
                        f"{float(probs[i][0].item()):.6f}",
                        f"{float(probs[i][1].item()):.6f}",
                        f"{float(scaled_logits[i][0].item()):.6f}",
                        f"{float(scaled_logits[i][1].item()):.6f}",
                    ]
                )

    return rows


def write_rows(path: Path, header: List[str], rows: List[List[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temperature scaling calibration for a saved HF classifier.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--validation-file", default=str(VALIDATION_FILE))
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--tokenizer-name", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    validation_file = Path(args.validation_file)
    test_file = Path(args.test_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_output = output_dir / f"{args.prefix}_validation_predictions.csv"
    test_output = output_dir / f"{args.prefix}_test_predictions.csv"
    test_scaled_output = output_dir / f"{args.prefix}_test_predictions_scaled.csv"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")
    if not validation_file.exists():
        raise FileNotFoundError(f"Validation file not found: {validation_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    print(f"Using model: {model_dir}", flush=True)
    print(f"Using device: {DEVICE}", flush=True)
    print(f"Validation file: {validation_file}", flush=True)
    print(f"Test file: {test_file}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)

    print("Starting validation inference...", flush=True)
    val_texts, val_labels = read_csv_file(validation_file)
    val_logits, val_y, val_rows = infer(model, tokenizer, val_texts, val_labels, stage_name="validation")
    write_rows(
        validation_output,
        ["text", "label", "pred_label", "confidence", "prob_0", "prob_1", "logit_0", "logit_1"],
        val_rows,
    )
    print(f"Saved validation predictions -> {validation_output}", flush=True)

    scaler = TemperatureScaler().to(DEVICE)
    temperature = scaler.fit(val_logits.to(DEVICE), val_y.to(DEVICE))

    print("Starting test inference (before scaling)...", flush=True)
    test_texts, test_labels = read_csv_file(test_file)
    _, _, test_rows = infer(model, tokenizer, test_texts, test_labels, stage_name="test")
    write_rows(
        test_output,
        ["text", "label", "pred_label", "confidence", "prob_0", "prob_1", "logit_0", "logit_1"],
        test_rows,
    )
    print(f"Saved test predictions -> {test_output}", flush=True)

    print("Starting test inference (after scaling)...", flush=True)
    scaled_rows = scaled_infer(
        model,
        tokenizer,
        test_texts,
        test_labels,
        temperature,
        stage_name="scaled test",
    )
    write_rows(
        test_scaled_output,
        ["text", "label", "pred_label", "confidence", "prob_0", "prob_1", "scaled_logit_0", "scaled_logit_1"],
        scaled_rows,
    )
    print(f"Saved scaled test predictions -> {test_scaled_output}", flush=True)

    print("Done. Next step: compute accuracy and calibration metrics from the saved CSV files.", flush=True)


if __name__ == "__main__":
    main()

