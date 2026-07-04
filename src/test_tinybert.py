from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "huawei-noah/TinyBERT_General_4L_312D"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

print("TinyBERT loaded successfully!")