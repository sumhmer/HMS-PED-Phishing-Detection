"""
Step 1: Data preparation.

Merges the two public email corpora (Ling and CEAS-2008), cleans them,
removes duplicates, and creates a stratified 70/15/15
train / validation / test split.

Label convention: 1 = phishing/malicious, 0 = legitimate.

Run:  python src/data_prep.py
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "splits"
SEED = 42
MAX_BODY_CHARS = 5000  # very long bodies add little signal and bloat the files


def clean_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load() -> pd.DataFrame:
    ling = pd.read_csv(RAW / "Ling.csv.gz")
    ling["source"] = "ling"
    ling["sender"] = ""

    ceas = pd.read_csv(RAW / "CEAS_08.csv.gz")
    ceas["source"] = "ceas"

    cols = ["sender", "subject", "body", "label", "source"]
    df = pd.concat([ling[cols], ceas[cols]], ignore_index=True)

    for c in ["sender", "subject", "body"]:
        df[c] = df[c].map(clean_text)
    df["body"] = df["body"].str.slice(0, MAX_BODY_CHARS)
    df["label"] = df["label"].astype(int)

    before = len(df)
    df = df[df["body"].str.len() > 0]
    df = df.drop_duplicates(subset=["subject", "body"]).reset_index(drop=True)
    print(f"Loaded {before} emails, {len(df)} after removing empty bodies and duplicates")
    return df


def split(df: pd.DataFrame):
    # Stratify on label AND source so every split has the same mix
    strata = df["label"].astype(str) + "_" + df["source"]
    train, temp = train_test_split(df, test_size=0.30, random_state=SEED, stratify=strata)
    temp_strata = temp["label"].astype(str) + "_" + temp["source"]
    val, test = train_test_split(temp, test_size=0.50, random_state=SEED, stratify=temp_strata)
    return train, val, test


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    train, val, test = split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(OUT / f"{name}.csv.gz", index=False)
        counts = part["label"].value_counts().to_dict()
        print(f"{name:5s}: {len(part):6d} emails | phishing={counts.get(1, 0)} legit={counts.get(0, 0)}")


if __name__ == "__main__":
    main()
