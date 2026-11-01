"""
Experiment 3: the LLM (Gemini) as a detector -- run this on YOUR machine.

It needs a Gemini API key (free from Google AI Studio). Put it in a file named
`.env` in the project root:   GEMINI_API_KEY=your_key_here
Never commit .env to GitHub (it is already listed in .gitignore).

Three evaluations:
  A) standalone : Gemini classifies a balanced random sample of test emails
                  (default 100), compared with the Stacking ensemble on the same emails.
  B) hybrid     : Gemini acts as a second-stage checker. Only emails that Stage 1
                  called legitimate but with an uncertain score (phishing probability
                  between --low and 0.5) are sent to Gemini; if Gemini says phishing,
                  the email is re-flagged. Metrics are reported on the full test split.
  C) cross      : the unseen-corpus test. A balanced sample of Ling emails (default 100)
                  is classified by Gemini and compared with the ensemble trained only on
                  CEAS-2008 (it never saw a Ling email). Run src/cross_corpus.py first.

Every answer is cached in results/gemini_cache.jsonl, so if you hit a rate limit
just run the same command again and it continues where it stopped.

Usage:
  pip install google-genai python-dotenv pandas scikit-learn
  python src/gemini_eval.py --mode standalone --n 100
  python src/gemini_eval.py --mode hybrid
  python src/gemini_eval.py --mode cross --n 100
Options: --model gemini-2.5-flash  --delay 4.5  (seconds between calls)
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, ROOT, SEED, load_split, metrics, save_json  # noqa: E402

# Same prompt as the original browser-extension backend (CMPE209FinalProject),
# so the offline evaluation matches what the extension does live.
PROMPT = """
You are an expert in phishing email detection.

Analyze the following email and respond in this JSON format:
{{
"is_phishing": true or false,
"explanation": "A short explanation of why you classified it that way."
}}

Email:
Subject: {subject}
From: {sender}
Body: {body}
"""
CACHE = RESULTS / "gemini_cache.jsonl"


def load_cache():
    cache = {}
    if CACHE.exists():
        for line in CACHE.open(encoding="utf-8"):
            rec = json.loads(line)
            cache[rec["key"]] = rec
    return cache


def parse(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    obj = json.loads(m.group(0) if m else text)
    return 1 if obj.get("is_phishing") in (True, "true", "True", 1) else 0, obj.get("explanation", "")


class Gemini:
    def __init__(self, model):
        from dotenv import load_dotenv
        from google import genai
        from google.genai import types
        load_dotenv(ROOT / ".env")
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY not found. Create a .env file in the project root.")
        self.client, self.types, self.model = genai.Client(api_key=key), types, model
        self.use_thinking_cfg = True

    def ask(self, row):
        prompt = PROMPT.format(subject=row["subject"], sender=row.get("sender", ""),
                               body=str(row["body"])[:4000])
        kw = dict(max_output_tokens=512, temperature=0, response_mime_type="application/json")
        if self.use_thinking_cfg:
            kw["thinking_config"] = self.types.ThinkingConfig(thinking_budget=0)
        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=prompt, config=self.types.GenerateContentConfig(**kw))
        except Exception as e:  # older models do not accept a thinking config
            if self.use_thinking_cfg and "thinking" in str(e).lower():
                self.use_thinking_cfg = False
                return self.ask(row)
            raise
        return resp.text


def classify(rows, llm, delay, split="test", max_retries=5):
    cache = load_cache()
    preds = []
    with CACHE.open("a", encoding="utf-8") as f:
        for i, (idx, row) in enumerate(rows.iterrows()):
            key = f"{llm.model}|{split}|{idx}"
            if key not in cache:
                for attempt in range(max_retries):
                    try:
                        raw = llm.ask(row)
                        pred, why = parse(raw)
                        break
                    except json.JSONDecodeError:
                        pred, why = 0, "UNPARSEABLE: " + str(raw)[:200]
                        break
                    except Exception as e:
                        wait = delay * (2 ** attempt) + 5
                        print(f"  error ({e.__class__.__name__}: {str(e)[:120]}), retrying in {wait:.0f}s")
                        time.sleep(wait)
                else:
                    sys.exit("Too many errors (rate limit or bad key/model). Run again later; progress is saved.")
                cache[key] = {"key": key, "pred": pred, "explanation": why}
                f.write(json.dumps(cache[key]) + "\n")
                f.flush()
                time.sleep(delay)
            preds.append(cache[key]["pred"])
            print(f"[{i + 1}/{len(rows)}] label={row['label']} gemini={preds[-1]}")
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["standalone", "hybrid", "cross"], required=True)
    ap.add_argument("--n", type=int, default=100, help="sample size for standalone and cross modes")
    ap.add_argument("--low", type=float, default=0.05, help="lower bound of uncertain band")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--delay", type=float, default=4.5)
    args = ap.parse_args()

    test = load_split("test")
    s1 = pd.read_csv(RESULTS / "stage1_test_predictions.csv")
    llm = Gemini(args.model)
    out_file = RESULTS / "gemini_results.json"
    out = json.load(open(out_file)) if out_file.exists() else {}

    if args.mode == "standalone":
        half = args.n // 2
        idx = np.concatenate([
            test[test.label == 1].sample(half, random_state=SEED).index,
            test[test.label == 0].sample(args.n - half, random_state=SEED).index])
        rows = test.loc[idx]
        g = classify(rows, llm, args.delay)
        out["standalone"] = {"model": args.model, "n": int(len(rows)),
                             "gemini": metrics(rows["label"], g),
                             "stacking_same_emails": metrics(rows["label"], s1.loc[idx, "stack_pred"]),
                             "hybrid_clkmeans_same_emails": metrics(rows["label"], s1.loc[idx, "hybrid_pred"])}
    elif args.mode == "cross":
        every = pd.concat([load_split(n) for n in ("train", "val", "test")], ignore_index=True)
        cp = pd.read_csv(RESULTS / "cross_ceas_to_ling_predictions.csv").set_index("row")
        ling = every.loc[cp.index]
        half = args.n // 2
        idx = np.concatenate([
            ling[ling.label == 1].sample(half, random_state=SEED).index,
            ling[ling.label == 0].sample(args.n - half, random_state=SEED).index])
        rows = every.loc[idx]
        g = classify(rows, llm, args.delay, split="all")
        out["cross"] = {"model": args.model, "n": int(len(rows)), "direction": "CEAS -> Ling",
                        "gemini": metrics(rows["label"], g),
                        "stacking_same_emails": metrics(rows["label"], cp.loc[idx, "stack_pred"]),
                        "hybrid_clkmeans_same_emails": metrics(rows["label"], cp.loc[idx, "hybrid_pred"]),
                        "stacking_or_gemini_same_emails": metrics(rows["label"],
                                                                  np.maximum(cp.loc[idx, "stack_pred"].values, g))}
    else:
        band = (s1["stack_pred"] == 0) & (s1["stack_proba"] >= args.low)
        rows = test.loc[band[band].index]
        print(f"{len(rows)} uncertain emails will be sent to Gemini")
        g = classify(rows, llm, args.delay)
        final = s1["stack_pred"].values.copy()
        final[rows.index] = np.maximum(final[rows.index], g)
        out["hybrid"] = {"model": args.model, "band_low": args.low, "n_sent_to_llm": int(len(rows)),
                         "stage1_stacking": metrics(test["label"], s1["stack_pred"]),
                         "hybrid_stacking_plus_gemini": metrics(test["label"], final),
                         "llm_flips": int(g.sum()),
                         "llm_correct_flips": int(((g == 1) & (rows["label"].values == 1)).sum())}
    save_json(out, "gemini_results.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
