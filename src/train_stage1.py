"""
Stage 1: supervised "signature-based" detector (adapted from HMS-IDS).

Four base learners -- Naive Bayes, Decision Tree, KNN, Logistic Regression --
are tuned on the validation split, then combined with a stacking ensemble.
Following HMS-IDS, the meta-learner is the best-performing base algorithm.

Run:  python src/train_stage1.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from common import MODELS, RESULTS, SEED, Timer, load_split, metrics, save_json
from features import FeatureBuilder

GRIDS = {
    "Naive Bayes": (MultinomialNB, [{"alpha": a} for a in (0.01, 0.1, 1.0)]),
    "Decision Tree": (DecisionTreeClassifier,
                      [{"max_depth": d, "random_state": SEED} for d in (20, 40, None)]),
    "KNN": (KNeighborsClassifier,
            [{"n_neighbors": k, "metric": "cosine", "algorithm": "brute"}
             for k in (3, 5, 11)]),
    "Logistic Regression": (LogisticRegression,
                            [{"C": c, "solver": "liblinear", "max_iter": 2000}
                             for c in (1.0, 10.0, 100.0)]),
}


def main():
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    y_tr, y_va, y_te = train["label"].values, val["label"].values, test["label"].values
    log = {}

    # 1) Choose how many IG-selected terms to keep, using the validation split
    k_scores = {}
    for k in (1000, 3000, 5000):
        fb = FeatureBuilder(k_best=k)
        Xtr, Xva = fb.fit_transform(train, y_tr), fb.transform(val)
        lr = LogisticRegression(C=10.0, solver="liblinear", max_iter=2000).fit(Xtr, y_tr)
        k_scores[k] = round(f1_score(y_va, lr.predict(Xva)), 4)
        print(f"IG top-k={k}: val F1={k_scores[k]}")
    best_k = max(k_scores, key=k_scores.get)
    log["ig_k_search"] = k_scores
    log["ig_k_best"] = best_k

    with Timer() as t_feat:
        fb = FeatureBuilder(k_best=best_k)
        Xtr = fb.fit_transform(train, y_tr)
    Xva, Xte = fb.transform(val), fb.transform(test)
    log["feature_time_s"] = t_feat.seconds
    log["n_features"] = Xtr.shape[1]
    log["top_ig_terms"] = [(t, round(float(s), 4)) for t, s in fb.top_terms(25)]

    # 2) Tune each base learner on validation
    best = {}
    log["tuning"] = {}
    for name, (cls, grid) in GRIDS.items():
        scores = []
        for params in grid:
            m = cls(**params).fit(Xtr, y_tr)
            scores.append((f1_score(y_va, m.predict(Xva)), params))
        f1v, params = max(scores, key=lambda s: s[0])
        best[name] = cls(**params)
        log["tuning"][name] = {"best_params": {k: v for k, v in params.items() if k != "n_jobs"},
                               "val_f1": round(f1v, 4),
                               "all": [(round(s, 4), {k: v for k, v in p.items() if k != "n_jobs"})
                                       for s, p in scores]}
        print(f"{name}: best {params} val F1={f1v:.4f}")

    # 3) Final training, 5-fold CV on train, and test evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted, probas = {}, {}, {}
    for name, model in best.items():
        cv_acc = cross_val_score(clone(model), Xtr, y_tr, cv=cv, scoring="accuracy")
        with Timer() as tt:
            model.fit(Xtr, y_tr)
        with Timer() as tp:
            pred = model.predict(Xte)
        r = metrics(y_te, pred)
        r.update(cv_mean=round(cv_acc.mean(), 4), cv_std=round(cv_acc.std(), 4),
                 train_time_s=tt.seconds, test_time_s=tp.seconds,
                 val_f1=log["tuning"][name]["val_f1"])
        results[name] = r
        fitted[name] = model
        probas[name] = model.predict_proba(Xte)[:, 1]
        print(name, r)

    # 4) Stacking ensemble; meta-learner = best base algorithm on validation (as in HMS-IDS)
    meta_name = max(log["tuning"], key=lambda n: log["tuning"][n]["val_f1"])
    meta_cls, _ = GRIDS[meta_name]
    meta_params = {"Logistic Regression": {"solver": "liblinear", "max_iter": 2000},
                   "Decision Tree": {"max_depth": 5, "random_state": SEED},
                   "KNN": {"n_neighbors": 15},
                   "Naive Bayes": {}}[meta_name]
    stack = StackingClassifier(
        estimators=[(n.replace(" ", "_"), clone(m)) for n, m in best.items()],
        final_estimator=meta_cls(**meta_params),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        stack_method="predict_proba")
    with Timer() as tt:
        stack.fit(Xtr, y_tr)
    with Timer() as tp:
        pred = stack.predict(Xte)
    r = metrics(y_te, pred)
    r.update(train_time_s=tt.seconds, test_time_s=tp.seconds,
             val_f1=round(f1_score(y_va, stack.predict(Xva)), 4), meta_learner=meta_name)
    results["Stacking Ensemble"] = r
    print("Stacking", r)

    log["results_test"] = results
    save_json(log, "stage1_results.json")
    pd.DataFrame(results).T.to_csv(RESULTS / "stage1_results.csv")

    # Save predictions for stage 2 and figures
    out = pd.DataFrame({"label": y_te, "stack_pred": pred,
                        "stack_proba": stack.predict_proba(Xte)[:, 1],
                        **{f"proba_{n}": p for n, p in probas.items()}})
    out.to_csv(RESULTS / "stage1_test_predictions.csv", index=False)
    pd.DataFrame({"label": y_va, "stack_pred": stack.predict(Xva),
                  "stack_proba": stack.predict_proba(Xva)[:, 1]}).to_csv(
        RESULTS / "stage1_val_predictions.csv", index=False)

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"features": fb, "stack": stack, "base": fitted}, MODELS / "stage1.joblib", compress=3)
    print("Saved models/stage1.joblib")


if __name__ == "__main__":
    main()
