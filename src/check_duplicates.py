import pandas as pd

train_df = pd.read_csv("data/train_phishing_legit.csv")
test_df = pd.read_csv("data/test_phishing_legit.csv")

train_texts = set(train_df["text"])
test_texts = set(test_df["text"])

overlap = train_texts.intersection(test_texts)

print("Train size:", len(train_texts))
print("Test size:", len(test_texts))
print("Duplicate emails:", len(overlap))