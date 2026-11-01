"""Shared helpers: loading splits, metrics, saving results."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
SEED = 42


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(SPLITS / f"{name}.csv.gz", keep_default_na=False)
    df["label"] = df["label"].astype(int)
    return df


def metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "fpr": round(fp / max(fp + tn, 1), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = round(time.perf_counter() - self.t, 3)


def save_json(obj, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
