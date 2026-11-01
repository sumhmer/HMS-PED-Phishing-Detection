"""
Writes the paper's result tables and numbers straight from results/*.json,
so no number in the paper is typed by hand.

Outputs (in paper/):  generated_numbers.tex, table_stage1.tex,
                      table_cross_corpus.tex, table_dataset.tex,
                      table_gemini.tex (only if results/gemini_results.json exists)
Run:  python src/export_latex.py
"""
import json

import pandas as pd

from common import RESULTS, ROOT, load_split

PAPER = ROOT / "paper"


def pct(x):
    return f"{100 * x:.2f}"


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main():
    PAPER.mkdir(exist_ok=True)
    s1 = json.load(open(RESULTS / "stage1_results.json"))
    s2 = json.load(open(RESULTS / "stage2_results.json"))
    cc = json.load(open(RESULTS / "cross_corpus_results.json"))
    gem_path = RESULTS / "gemini_results.json"
    gem = json.load(open(gem_path)) if gem_path.exists() else {}
    splits = {n: load_split(n) for n in ("train", "val", "test")}

    # ---------- dataset table ----------
    rows = []
    for n, df in splits.items():
        rows.append((n.capitalize() if n != "val" else "Validation", len(df),
                     int((df.label == 1).sum()), int((df.label == 0).sum())))
    total = sum(r[1] for r in rows)
    with open(PAPER / "table_dataset.tex", "w") as f:
        f.write("\\begin{tabular}{lrrr}\n\\toprule\nSplit & Emails & Phishing & Legitimate\\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r[0]} & {r[1]:,} & {r[2]:,} & {r[3]:,}\\\\\n")
        f.write(f"\\midrule\nTotal & {total:,} & {sum(r[2] for r in rows):,} & {sum(r[3] for r in rows):,}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------- stage-1 / hybrid table ----------
    res = dict(s1["results_test"])
    res["CL-k-means (alone)"] = s2["clkmeans_alone_test"]
    res["\\textbf{Hybrid (proposed)}"] = s2["hybrid_test"]
    best_f1 = max(v["f1"] for v in res.values())
    with open(PAPER / "table_stage1.tex", "w") as f:
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n"
                "Model & CV Acc. & Acc. (\\%) & Prec. & Rec. & F1 & FPR\\\\\n\\midrule\n")
        for name, v in res.items():
            cv = f"{v['cv_mean']:.4f}" if "cv_mean" in v else "--"
            f1 = f"\\textbf{{{v['f1']:.4f}}}" if v["f1"] == best_f1 else f"{v['f1']:.4f}"
            if name == "CL-k-means (alone)" or name.startswith("\\textbf"):
                f.write("\\midrule\n" if name == "CL-k-means (alone)" else "")
            f.write(f"{name} & {cv} & {pct(v['accuracy'])} & {v['precision']:.4f} & "
                    f"{v['recall']:.4f} & {f1} & {v['fpr']:.4f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------- cross-corpus table ----------
    models = ["Naive Bayes", "Decision Tree", "KNN", "Logistic Regression", "Stacking Ensemble",
              "CL-k-means alone", "Hybrid (Stacking + CL-k-means)"]
    dirs = list(cc)
    with open(PAPER / "table_cross_corpus.tex", "w") as f:
        f.write("\\begin{tabular}{l" + "cc" * len(dirs) + "}\n\\toprule\n")
        f.write(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{d.replace('->', '$\\rightarrow$')}}}" for d in dirs) + "\\\\\n")
        f.write(" & " + " & ".join("Acc. & F1" for _ in dirs) + "\\\\\n\\midrule\n")
        for m in models:
            label = "\\textbf{Hybrid (proposed)}" if m.startswith("Hybrid") else m
            f.write(label + " & " + " & ".join(f"{cc[d][m]['accuracy']:.4f} & {cc[d][m]['f1']:.4f}" for d in dirs) + "\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------- tau sensitivity ----------
    sens_in = s2["tau_sensitivity_test"]
    sens_x = cc["CEAS -> Ling"]["_tau_sensitivity"]
    with open(PAPER / "table_tau.tex", "w") as f:
        f.write("\\begin{tabular}{ccccccc}\n\\toprule\n"
                " & \\multicolumn{3}{c}{In-distribution test} & \\multicolumn{3}{c}{CEAS$\\rightarrow$Ling}\\\\\n"
                "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
                "$\\tau$ & Rec. & FPR & F1 & Rec. & FPR & F1\\\\\n\\midrule\n")
        st_in, st_x = s1["results_test"]["Stacking Ensemble"], cc["CEAS -> Ling"]["Stacking Ensemble"]
        f.write(f"Stage 1 only & {st_in['recall']:.4f} & {st_in['fpr']:.4f} & {st_in['f1']:.4f} & "
                f"{st_x['recall']:.4f} & {st_x['fpr']:.4f} & {st_x['f1']:.4f}\\\\\n")
        for t in sens_in:
            a, b = sens_in[t], sens_x[t]
            f.write(f"{t} & {a['recall']:.4f} & {a['fpr']:.4f} & {a['f1']:.4f} & "
                    f"{b['recall']:.4f} & {b['fpr']:.4f} & {b['f1']:.4f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------- inline numbers ----------
    st, hy, ck = res["Stacking Ensemble"], s2["hybrid_test"], s2["clkmeans_alone_test"]
    lr = res["Logistic Regression"]
    m = ""
    m += macro("NTotal", f"{total:,}")
    m += macro("NTrain", f"{len(splits['train']):,}")
    m += macro("NVal", f"{len(splits['val']):,}")
    m += macro("NTest", f"{len(splits['test']):,}")
    m += macro("IGk", f"{s1['ig_k_best']:,}")
    m += macro("NFeat", f"{s1['n_features']:,}")
    m += macro("MetaLearner", st["meta_learner"])
    m += macro("StackAcc", pct(st["accuracy"]))
    m += macro("StackFone", f"{st['f1']:.4f}")
    m += macro("StackFPR", pct(st["fpr"]))
    m += macro("StackFN", str(st["fn"]))
    m += macro("StackFP", str(st["fp"]))
    m += macro("LRAcc", pct(lr["accuracy"]))
    m += macro("HybAcc", pct(hy["accuracy"]))
    m += macro("HybFone", f"{hy['f1']:.4f}")
    m += macro("HybRec", f"{hy['recall']:.4f}")
    m += macro("HybFPR", pct(hy["fpr"]))
    m += macro("HybFN", str(hy["fn"]))
    m += macro("HybFP", str(hy["fp"]))
    m += macro("CLKAcc", pct(ck["accuracy"]))
    m += macro("CLKk", str(s2["chosen_k"]))
    m += macro("CLKtau", str(s2["chosen_tau"]))
    m += macro("StageTwoFlips", str(s2["stage2_flips"]))
    m += macro("StageTwoCorrect", str(s2["stage2_correct_flips"]))
    m += macro("StageOneMissed", str(s2["stage1_missed_phishing"]))
    m += macro("StackTrainTime", f"{st['train_time_s']:.1f}")
    m += macro("StackTestTime", f"{st['test_time_s']:.1f}")
    for d, tag in zip(dirs, ["CL", "LC"]):
        for key, mm in [("Stack", "Stacking Ensemble"), ("Hyb", "Hybrid (Stacking + CL-k-means)"),
                        ("CLK", "CL-k-means alone")]:
            m += macro(f"X{tag}{key}Fone", f"{cc[d][mm]['f1']:.4f}")
            m += macro(f"X{tag}{key}Rec", f"{cc[d][mm]['recall']:.4f}")
            m += macro(f"X{tag}{key}Acc", pct(cc[d][mm]['accuracy']))
        m += macro(f"X{tag}Ntest", f"{cc[d]['_n_test']:,}")
        m += macro(f"X{tag}Ntrain", f"{cc[d]['_n_train']:,}")
    if "standalone" in gem:
        g = gem["standalone"]
        m += macro("GemModel", g["model"])
        m += macro("GemN", str(g["n"]))
        m += macro("GemAcc", pct(g["gemini"]["accuracy"]))
        m += macro("GemFone", f"{g['gemini']['f1']:.4f}")
    if "cross" in gem:
        g = gem["cross"]
        m += macro("GemXN", str(g["n"]))
        m += macro("GemXFone", f"{g['gemini']['f1']:.4f}")
        m += macro("GemXRec", f"{g['gemini']['recall']:.4f}")
        m += macro("GemXStackFone", f"{g['stacking_same_emails']['f1']:.4f}")
        m += macro("GemXOrFone", f"{g['stacking_or_gemini_same_emails']['f1']:.4f}")
    every = pd.concat(list(splits.values()))
    ling = every[every.source == "ling"]
    m += macro("LingMajority", pct((ling.label == 0).mean()))
    m += macro("HaveGemini", "1" if gem else "0")
    lo = s2["tau_sensitivity_test"]["0.8"]
    m += macro("TauEightRec", f"{lo['recall']:.4f}")
    m += macro("TauEightFPR", pct(lo["fpr"]))
    m += macro("TauEightFN", str(lo["fn"]))
    (PAPER / "generated_numbers.tex").write_text(m)

    if gem:
        with open(PAPER / "table_gemini.tex", "w") as f:
            f.write("\\begin{tabular}{lcccc}\n\\toprule\nSystem & Acc. (\\%) & Prec. & Rec. & F1\\\\\n\\midrule\n")
            if "standalone" in gem:
                g = gem["standalone"]
                for lab, key in [(f"Gemini ({g['model']}) zero-shot", "gemini"),
                                 ("Stacking (same emails)", "stacking_same_emails"),
                                 ("Hybrid (same emails)", "hybrid_clkmeans_same_emails")]:
                    v = g[key]
                    f.write(f"{lab} & {pct(v['accuracy'])} & {v['precision']:.4f} & {v['recall']:.4f} & {v['f1']:.4f}\\\\\n")
            if "cross" in gem:
                g = gem["cross"]
                f.write("\\midrule\n\\multicolumn{5}{l}{\\emph{Unseen corpus (CEAS$\\rightarrow$Ling), " + str(g["n"]) + " emails}}\\\\\n")
                for lab, key in [(f"Gemini ({g['model']}) zero-shot", "gemini"),
                                 ("Stacking (trained on CEAS)", "stacking_same_emails"),
                                 ("Hybrid CL-k-means (CEAS)", "hybrid_clkmeans_same_emails"),
                                 ("Stacking OR Gemini", "stacking_or_gemini_same_emails")]:
                    v = g[key]
                    f.write(f"{lab} & {pct(v['accuracy'])} & {v['precision']:.4f} & {v['recall']:.4f} & {v['f1']:.4f}\\\\\n")
            if "hybrid" in gem:
                g = gem["hybrid"]
                f.write("\\midrule\n")
                for lab, key in [("Stacking (full test)", "stage1_stacking"),
                                 ("Stacking + Gemini (full test)", "hybrid_stacking_plus_gemini")]:
                    v = g[key]
                    f.write(f"{lab} & {pct(v['accuracy'])} & {v['precision']:.4f} & {v['recall']:.4f} & {v['f1']:.4f}\\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    print("LaTeX tables and numbers written to", PAPER)


if __name__ == "__main__":
    main()
