import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "data/Phishing Email Dataset/phishing_email.csv"
)

# Development sample
df = df.sample(n=5000, random_state=42)

# Rename column to match EduPhish
df = df.rename(columns={"text_combined": "text"})

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Train:", train_df.shape)
print("Test:", test_df.shape)

train_df.to_csv(
    "data/train_phishing_email.csv",
    index=False
)

test_df.to_csv(
    "data/test_phishing_email.csv",
    index=False
)