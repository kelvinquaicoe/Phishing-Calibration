from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

print("RoBERTa loaded successfully!")