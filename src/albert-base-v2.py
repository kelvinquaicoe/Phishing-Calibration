from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "albert-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

print("ALBERT loaded successfully!")