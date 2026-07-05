import pandas as pd
from pathlib import Path

out = Path("data/eval_subsets")
out.mkdir(parents=True, exist_ok=True)

datasets = {
    "ceas08": "data/Phishing Email Dataset/CEAS_08.csv",
    "enron": "data/Phishing Email Dataset/Enron.csv",
    "spamassassin": "data/Phishing Email Dataset/SpamAssasin.csv",
}

for name, path in datasets.items():
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    text_col = "text" if "text" in df.columns else "body" if "body" in df.columns else "text_combined"
    df = df[[text_col, "label"]].dropna().rename(columns={text_col: "text"})

    n = min(2500, df["label"].value_counts().min())
    parts = []
    for label_value in sorted(df["label"].unique()):
        parts.append(df[df["label"] == label_value].sample(n=n, random_state=42))

    sampled = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

    outfile = out / f"{name}_eval_5000.csv"
    sampled.to_csv(outfile, index=False)
    print(outfile, sampled.shape, sampled["label"].value_counts().to_dict())
