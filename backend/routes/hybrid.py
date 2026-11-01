"""
/api/hybrid-scan : the full hybrid detector used by the browser extension.

  Stage 1  Stacking ensemble (NB, DT, KNN, LR)   -> phishing probability
  Stage 2  CL-k-means anomaly layer              -> re-checks emails called legitimate
  Stage 3  Gemini LLM (optional)                 -> re-checks uncertain emails and
                                                    writes a plain-language explanation

Models are produced by src/train_stage1.py and src/stage2_clkmeans.py.
"""
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Blueprint, jsonify, request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))          # so the pickled FeatureBuilder can be loaded

from llm_client import ask_gemini  # noqa: E402

hybrid_bp = Blueprint("hybrid", __name__)
UNCERTAIN_LOW = float(os.getenv("UNCERTAIN_LOW", "0.05"))
_models = {}


def models():
    if not _models:
        s1 = joblib.load(ROOT / "models" / "stage1.joblib")
        s2 = joblib.load(ROOT / "models" / "stage2.joblib")
        _models.update(features=s1["features"], stack=s1["stack"],
                       clk=s2["clkmeans"], tau=s2["tau"])
    return _models


@hybrid_bp.route("/api/hybrid-scan", methods=["POST"])
def hybrid_scan():
    data = request.json or {}
    email = {"sender": data.get("sender", ""), "subject": data.get("subject", ""),
             "body": data.get("body", "")}
    if not email["body"].strip():
        return jsonify({"error": "Email body is empty"}), 400

    m = models()
    X = m["features"].transform(pd.DataFrame([email]))
    proba = float(m["stack"].predict_proba(X)[0, 1])
    is_phishing, stage = proba >= 0.5, "Stage 1 (stacking ensemble)"

    if not is_phishing:
        cl_label, purity = m["clk"].assign(X)
        if cl_label[0] == 1 and purity[0] >= m["tau"]:
            is_phishing, stage = True, "Stage 2 (CL-k-means anomaly layer)"

    explanation, llm_used = None, False
    uncertain = (not is_phishing) and proba >= UNCERTAIN_LOW
    if os.getenv("GEMINI_API_KEY") and (uncertain or data.get("explain", True)):
        try:
            llm = ask_gemini(email)
            llm_used = True
            explanation = llm.get("explanation")
            if uncertain and llm.get("is_phishing"):
                is_phishing, stage = True, "Stage 3 (Gemini LLM check)"
            elif bool(llm.get("is_phishing")) != is_phishing:
                explanation = "Note: the LLM's own verdict differs from the ML verdict. " + (explanation or "")
        except Exception as e:  # the ML verdict still stands without the LLM
            explanation = f"(LLM unavailable: {e.__class__.__name__})"

    return jsonify({
        "is_phishing": bool(is_phishing),
        "phishing_probability": round(proba, 3),
        "decided_by": stage,
        "llm_used": llm_used,
        "explanation": explanation or ("High phishing probability from the ML ensemble."
                                       if is_phishing else "No phishing indicators found by the ML stages."),
    })
