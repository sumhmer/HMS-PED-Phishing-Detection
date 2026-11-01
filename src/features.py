"""
Feature engineering for the hybrid phishing detector.

Two kinds of features, mirroring the HMS-IDS idea of combining raw data with
engineered features:
  1. TF-IDF word / bigram features from subject + body, reduced with
     Information Gain (IG) feature selection (same criterion as HMS-IDS).
  2. A small set of hand-crafted phishing indicators (links, urgency words, ...).
"""
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

URL_RE = re.compile(r"(https?://|www\.)\S+", re.I)
URGENCY = ["urgent", "immediately", "verify", "suspend", "suspended", "confirm",
           "password", "account", "login", "log in", "click", "update", "bank",
           "security", "limited time", "act now", "winner", "prize", "expire"]
HAND_FEATURES = ["n_urls", "has_url", "n_exclaim", "upper_ratio", "log_len",
                 "n_money", "n_urgency", "subj_upper_ratio", "n_ip_urls"]


def email_text(df: pd.DataFrame) -> pd.Series:
    return df["subject"].fillna("") + " " + df["body"].fillna("")


def hand_crafted(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for subj, body in zip(df["subject"].fillna(""), df["body"].fillna("")):
        low = body.lower()
        urls = URL_RE.findall(body)
        letters = [c for c in body if c.isalpha()]
        s_letters = [c for c in subj if c.isalpha()]
        rows.append([
            len(urls),
            1.0 if urls else 0.0,
            body.count("!"),
            sum(c.isupper() for c in letters) / max(len(letters), 1),
            np.log1p(len(body)),
            body.count("$") + body.count("€") + body.count("£"),
            sum(low.count(w) for w in URGENCY),
            sum(c.isupper() for c in s_letters) / max(len(s_letters), 1),
            len(re.findall(r"https?://\d+\.\d+\.\d+\.\d+", body)),
        ])
    return np.asarray(rows, dtype=float)


def information_gain(X_bin: sparse.csr_matrix, y: np.ndarray) -> np.ndarray:
    """IG(Y; term) = H(Y) - H(Y | term present/absent), for every column."""
    y = np.asarray(y)
    n = len(y)

    def H(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    h_y = H(y.mean())
    present = np.asarray(X_bin.sum(axis=0)).ravel()            # docs containing term
    present_pos = np.asarray(X_bin[y == 1].sum(axis=0)).ravel()
    absent = n - present
    absent_pos = y.sum() - present_pos
    p_present = present / n
    h_present = np.where(present > 0, H(present_pos / np.maximum(present, 1)), 0)
    h_absent = np.where(absent > 0, H(absent_pos / np.maximum(absent, 1)), 0)
    return h_y - (p_present * h_present + (1 - p_present) * h_absent)


class FeatureBuilder:
    def __init__(self, max_features=30000, k_best=3000):
        self.tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3,
                                     max_df=0.9, max_features=max_features,
                                     sublinear_tf=True, strip_accents="unicode")
        self.scaler = MaxAbsScaler()
        self.k_best = k_best
        self.selected = None
        self.ig_scores = None

    def fit(self, df, y):
        X = self.tfidf.fit_transform(email_text(df))
        self.ig_scores = information_gain((X > 0).astype(np.int8).tocsr(), y)
        self.selected = np.argsort(self.ig_scores)[::-1][: self.k_best]
        self.scaler.fit(hand_crafted(df))
        return self

    def transform(self, df):
        X = self.tfidf.transform(email_text(df))[:, self.selected]
        H = sparse.csr_matrix(self.scaler.transform(hand_crafted(df)))
        return sparse.hstack([X, H]).tocsr()

    def fit_transform(self, df, y):
        return self.fit(df, y).transform(df)

    def top_terms(self, n=20):
        vocab = np.array(self.tfidf.get_feature_names_out())
        idx = np.argsort(self.ig_scores)[::-1][:n]
        return list(zip(vocab[idx], self.ig_scores[idx]))
