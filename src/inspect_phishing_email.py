import pandas as pd

df = pd.read_csv(
    "data/Phishing Email Dataset/phishing_email.csv"
)

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())