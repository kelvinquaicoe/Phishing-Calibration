import pandas as pd
from sklearn.model_selection import train_test_split

# Load EduPhish dataset
df = pd.read_csv(
    "data/Education-Targeted Phishing Email Dataset/eduphish_dataset.csv"
)

# Take a random sample of 5000 emails
df = df.sample(n=5000, random_state=42)

print("Sample Size:", df.shape)

# Split into train and test sets
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Train:", train_df.shape)
print("Test:", test_df.shape)

# Save the split datasets
train_df.to_csv("data/train_eduphish.csv", index=False)
test_df.to_csv("data/test_eduphish.csv", index=False)

print("Files saved:")
print("data/train_eduphish.csv")
print("data/test_eduphish.csv")