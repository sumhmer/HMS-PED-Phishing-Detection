"""/api/llm-query : LLM-only scan (the original extension's behaviour), kept for comparison."""
from flask import Blueprint, jsonify, request

from llm_client import ask_gemini

llm_bp = Blueprint("llm", __name__)


@llm_bp.route("/api/llm-query", methods=["POST"])
def generate():
    data = request.json or {}
    if not data.get("body"):
        return jsonify({"error": "Missing email body"}), 400
    try:
        parsed = ask_gemini(data)
    except Exception as e:
        return jsonify({"error": f"LLM call failed: {e}"}), 500
    return jsonify({"is_phishing": parsed.get("is_phishing"),
                    "explanation": parsed.get("explanation")})
