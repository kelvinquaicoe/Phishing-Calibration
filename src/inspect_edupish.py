import pandas as pd

df = pd.read_csv(
    "data/Education-Targeted Phishing Email Dataset/eduphish_dataset.csv"
)

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nLabel Counts:")
print(df["label"].value_counts())

print("\nFirst 5 rows:")
print(df.head())