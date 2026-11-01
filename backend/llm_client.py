"""Gemini helper shared by the /api/llm-query and /api/hybrid-scan routes.

The prompt is the one written in the original CMPE209FinalProject backend.
Model is configurable with GEMINI_MODEL in .env (default gemini-2.5-flash,
because gemini-2.0-flash used by the original project is being retired).
"""
import json
import os
import re

from google import genai
from google.genai import types

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


def ask_gemini(email: dict) -> dict:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = PROMPT.format(subject=email.get("subject", ""), sender=email.get("sender", ""),
                           body=str(email.get("body", ""))[:4000])
    cfg = dict(max_output_tokens=512, temperature=0, response_mime_type="application/json")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        resp = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(**cfg, thinking_config=types.ThinkingConfig(thinking_budget=0)))
    except Exception as e:
        if "thinking" not in str(e).lower():
            raise
        resp = client.models.generate_content(model=model, contents=prompt,
                                              config=types.GenerateContentConfig(**cfg))
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (resp.text or "").strip())
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0) if match else text)
