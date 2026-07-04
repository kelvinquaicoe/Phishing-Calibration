import pandas as pd

df = pd.read_csv("data/phishing_legit_dataset_KD_10000.csv")

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())