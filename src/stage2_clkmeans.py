"""
Stage 2: unsupervised anomaly layer -- Cluster-Labelling k-means (CL-k-means),
adapted from HMS-IDS.

Emails that Stage 1 calls *legitimate* are passed to Stage 2. Each email is
mapped to its nearest k-means cluster (clusters learned on the training set and
labelled by majority vote). If that cluster is a phishing cluster AND its purity
(the HMS-IDS "confidence") is at least tau, the email is re-flagged as phishing.
k and tau are chosen on the validation split.

Run:  python src/stage2_clkmeans.py      (after train_stage1.py)
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

from common import MODELS, RESULTS, SEED, Timer, load_split, metrics, save_json

K_GRID = (50, 100, 200, 400)
TAU_GRID = (0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)


class CLKMeans:
    def __init__(self, k=200, n_components=100):
        self.k = k
        self.svd = TruncatedSVD(n_components=n_components, random_state=SEED)
        self.km = MiniBatchKMeans(n_clusters=k, random_state=SEED, batch_size=2048, n_init=3)

    def _embed(self, X, fit=False):
        Z = self.svd.fit_transform(X) if fit else self.svd.transform(X)
        return normalize(Z)

    def fit(self, X, y):
        Z = self._embed(X, fit=True)
        assign = self.km.fit_predict(Z)
        y = np.asarray(y)
        self.cluster_label = np.zeros(self.k, dtype=int)
        self.purity = np.zeros(self.k)
        for c in range(self.k):
            members = y[assign == c]
            if len(members) == 0:
                continue
            frac_pos = members.mean()
            self.cluster_label[c] = int(frac_pos >= 0.5)
            self.purity[c] = max(frac_pos, 1 - frac_pos)   # HMS-IDS confidence
        return self

    def assign(self, X):
        c = self.km.predict(self._embed(X))
        return self.cluster_label[c], self.purity[c]


def hybrid(stage1_pred, cl_label, cl_purity, tau):
    """Stage-2 only looks at emails Stage 1 called legitimate."""
    final = np.asarray(stage1_pred).copy()
    flip = (final == 0) & (cl_label == 1) & (cl_purity >= tau)
    final[flip] = 1
    return final, flip


def main():
    bundle = joblib.load(MODELS / "stage1.joblib")
    fb = bundle["features"]
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    Xtr, Xva, Xte = fb.transform(train), fb.transform(val), fb.transform(test)
    y_tr, y_va, y_te = train["label"].values, val["label"].values, test["label"].values
    s1_val = pd.read_csv(RESULTS / "stage1_val_predictions.csv")["stack_pred"].values
    s1_test = pd.read_csv(RESULTS / "stage1_test_predictions.csv")["stack_pred"].values

    search, models = [], {}
    for k in K_GRID:
        m = CLKMeans(k=k).fit(Xtr, y_tr)
        models[k] = m
        lab, pur = m.assign(Xva)
        alone_f1 = f1_score(y_va, lab)
        for tau in TAU_GRID:
            final, _ = hybrid(s1_val, lab, pur, tau)
            search.append({"k": k, "tau": tau, "val_f1_hybrid": round(f1_score(y_va, final), 4),
                           "val_f1_clkmeans_alone": round(alone_f1, 4)})
        print(f"k={k}: CL-k-means alone val F1={alone_f1:.4f}")
    search_df = pd.DataFrame(search)
    # k: the clustering that labels emails best on its own (validation F1);
    # tau: the purity threshold that gives the best hybrid validation F1 for that k
    #      (ties -> the stricter, larger tau)
    k = int(search_df.sort_values(["val_f1_clkmeans_alone", "k"], ascending=False).iloc[0]["k"])
    sub = search_df[search_df.k == k]
    tau = float(sub.sort_values(["val_f1_hybrid", "tau"], ascending=False).iloc[0]["tau"])
    print(f"Chosen k={k}, tau={tau}")

    m = models[k]
    with Timer() as t_fit:
        CLKMeans(k=k).fit(Xtr, y_tr)        # timing a clean refit
    with Timer() as t_pred:
        lab, pur = m.assign(Xte)
    final, flip = hybrid(s1_test, lab, pur, tau)
    sensitivity = {str(t): metrics(y_te, hybrid(s1_test, lab, pur, t)[0]) for t in TAU_GRID}

    out = {
        "tau_sensitivity_test": sensitivity,
        "chosen_k": k, "chosen_tau": tau,
        "search": search,
        "clkmeans_alone_test": {**metrics(y_te, lab), "train_time_s": t_fit.seconds,
                                "test_time_s": t_pred.seconds},
        "stage1_stacking_test": metrics(y_te, s1_test),
        "hybrid_test": metrics(y_te, final),
        "stage2_flips": int(flip.sum()),
        "stage2_correct_flips": int((flip & (y_te == 1)).sum()),
        "stage1_missed_phishing": int(((s1_test == 0) & (y_te == 1)).sum()),
        "n_clusters_phishing": int(m.cluster_label.sum()),
        "mean_purity": round(float(m.purity.mean()), 4),
    }
    for key in ("clkmeans_alone_test", "stage1_stacking_test", "hybrid_test"):
        print(key, out[key])
    print("flips:", out["stage2_flips"], "correct:", out["stage2_correct_flips"],
          "stage1 missed:", out["stage1_missed_phishing"])
    save_json(out, "stage2_results.json")
    search_df.to_csv(RESULTS / "stage2_search.csv", index=False)

    preds = pd.read_csv(RESULTS / "stage1_test_predictions.csv")
    preds = preds.drop(columns=[c for c in ("clk_label", "clk_purity", "hybrid_pred") if c in preds])
    preds["clk_label"], preds["clk_purity"], preds["hybrid_pred"] = lab, pur, final
    preds.to_csv(RESULTS / "stage1_test_predictions.csv", index=False)
    joblib.dump({"clkmeans": m, "tau": tau}, MODELS / "stage2.joblib", compress=3)


if __name__ == "__main__":
    # import through the module name so the saved model refers to
    # stage2_clkmeans.CLKMeans (loadable by the backend), not __main__.CLKMeans
    import stage2_clkmeans
    stage2_clkmeans.main()
