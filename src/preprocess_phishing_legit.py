import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "data/phishing_legit_dataset_KD_10000.csv"
)

# Development sample
df = df.sample(n=5000, random_state=42)

# Keep only the columns we need
df = df[["text", "label"]]

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Train:", train_df.shape)
print("Test:", test_df.shape)

train_df.to_csv(
    "data/train_phishing_legit.csv",
    index=False
)

test_df.to_csv(
    "data/test_phishing_legit.csv",
    index=False
)