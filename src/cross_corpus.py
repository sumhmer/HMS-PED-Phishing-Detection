"""
Experiment 2: "unseen phishing" (zero-day style) evaluation.

HMS-IDS separates known attacks from unknown / zero-day ones. The email analogue
is phishing from a campaign the model never saw. We simulate this by training on
one corpus and testing on the other:
  * CEAS-2008 -> Ling   (train on CEAS rows of the train split, test on every Ling email)
  * Ling -> CEAS-2008   (train on Ling rows of the train split, test on CEAS rows of the test split)
Hyper-parameters are reused from the in-distribution experiment (no re-tuning
on the target corpus).

Run:  python src/cross_corpus.py      (after train_stage1.py and stage2_clkmeans.py)
"""
import json

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from common import RESULTS, SEED, load_split, metrics, save_json
from features import FeatureBuilder
from stage2_clkmeans import CLKMeans, hybrid

CLASSES = {"Naive Bayes": MultinomialNB, "Decision Tree": DecisionTreeClassifier,
           "KNN": KNeighborsClassifier, "Logistic Regression": LogisticRegression}
META_PARAMS = {"Logistic Regression": {"solver": "liblinear", "max_iter": 2000},
               "Decision Tree": {"max_depth": 5, "random_state": SEED},
               "KNN": {"n_neighbors": 15}, "Naive Bayes": {}}


def run(train_df, test_df, s1, s2, save_as=None):
    y_tr, y_te = train_df["label"].values, test_df["label"].values
    fb = FeatureBuilder(k_best=s1["ig_k_best"])
    Xtr = fb.fit_transform(train_df, y_tr)
    Xte = fb.transform(test_df)

    base = {n: CLASSES[n](**s1["tuning"][n]["best_params"]) for n in CLASSES}
    res = {}
    for n, m in base.items():
        res[n] = metrics(y_te, clone(m).fit(Xtr, y_tr).predict(Xte))
    meta = s1["results_test"]["Stacking Ensemble"]["meta_learner"]
    stack = StackingClassifier([(n.replace(" ", "_"), clone(m)) for n, m in base.items()],
                               final_estimator=CLASSES[meta](**META_PARAMS[meta]),
                               cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                               stack_method="predict_proba").fit(Xtr, y_tr)
    s1_pred = stack.predict(Xte)
    res["Stacking Ensemble"] = metrics(y_te, s1_pred)

    clk = CLKMeans(k=s2["chosen_k"]).fit(Xtr, y_tr)
    lab, pur = clk.assign(Xte)
    res["CL-k-means alone"] = metrics(y_te, lab)
    final, flip = hybrid(s1_pred, lab, pur, s2["chosen_tau"])
    res["Hybrid (Stacking + CL-k-means)"] = metrics(y_te, final)
    res["_tau_sensitivity"] = {str(t): metrics(y_te, hybrid(s1_pred, lab, pur, t)[0])
                               for t in (0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)}
    res["_stage2_flips"] = int(flip.sum())
    res["_stage2_correct_flips"] = int((flip & (y_te == 1)).sum())
    res["_n_train"], res["_n_test"] = len(train_df), len(test_df)
    if save_as:   # per-email predictions, used by gemini_eval.py --mode cross
        pd.DataFrame({"row": test_df.index, "label": y_te, "stack_pred": s1_pred,
                      "stack_proba": stack.predict_proba(Xte)[:, 1], "hybrid_pred": final}
                     ).to_csv(RESULTS / save_as, index=False)
    return res


def main():
    s1 = json.load(open(RESULTS / "stage1_results.json"))
    s2 = json.load(open(RESULTS / "stage2_results.json"))
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    every = pd.concat([train, val, test], ignore_index=True)

    out = {
        "CEAS -> Ling": run(train[train.source == "ceas"], every[every.source == "ling"], s1, s2,
                            save_as="cross_ceas_to_ling_predictions.csv"),
        "Ling -> CEAS": run(train[train.source == "ling"], test[test.source == "ceas"], s1, s2),
    }
    for d, r in out.items():
        print("\n==", d)
        for k, v in r.items():
            if k != "_tau_sensitivity":
                print(f"  {k}: {v}")
        for t, v in r["_tau_sensitivity"].items():
            print(f"  tau={t}: acc={v['accuracy']} rec={v['recall']} fpr={v['fpr']} f1={v['f1']}")
    save_json(out, "cross_corpus_results.json")


if __name__ == "__main__":
    main()
