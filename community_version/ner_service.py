#!/usr/bin/env python3
"""
Synchronous NER

- Extracts entities with spaCy.

Endpoints
---------
GET  /health
POST /ner
    Request (JSON):
        {
          "text": "Your input text...",
          "labels": ["PERSON","ORG","GPE"]   # optional override of allowed labels
        }
    Response (JSON):
        {
          "text": "...",
          "model": "en_core_web_sm",
          "entities": ["openai", "san francisco"],
          "request_id": "..."
        }

Run
---
$ export SPACY_MODEL=en_core_web_sm
$ pip install flask spacy requests
$ python -m spacy download en_core_web_sm
$ python app.py
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

import requests
from flask import Flask, jsonify, request
import spacy
from functools import lru_cache

# ---------------------------
# Configuration helpers
# ---------------------------

def _as_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

def _scheme(ssl: bool) -> str:
    return "https" if ssl else "http"

def _base_url(host: str, port: int, ssl: bool) -> str:
    return f"{_scheme(ssl)}://{host}:{port}"

def _basic_auth(user: str, pwd: str):
    return (user, pwd) if (user and pwd) else None

# ---------------------------
# Environment
# ---------------------------

SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# HOT (dest) - how THIS service reaches it
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "127.0.0.1")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9201"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS", "")
OPENSEARCH_SSL = _as_bool(os.getenv("OPENSEARCH_SSL"), False)
OPENSEARCH_VERIFY_SSL = _as_bool(os.getenv("OPENSEARCH_VERIFY_SSL"), True)

# Default set of "interesting" entity types.
DEFAULT_INTERESTING_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "PRODUCT",
    "GPE",
    "EVENT",
    "WORK_OF_ART",
    "NORP",
    "LOC",
    "FAC",
}

# ---------------------------
# spaCy load
# ---------------------------

@lru_cache(maxsize=1)
def load_spacy():
    return spacy.load(SPACY_MODEL)

nlp = load_spacy()

# ---------------------------
# Entity Extraction
# ---------------------------

def _extract_entities(nlp_obj: spacy.Language, text: str, allowed_labels: set[str]) -> List[Tuple[str, str]]:
    doc = nlp_obj(text)
    return [
        (ent.text.strip().lower(), ent.label_)
        for ent in doc.ents
        if ent.label_ in allowed_labels and len(ent.text.strip()) >= 3
    ]

def _extract_normalized_entities(nlp_obj: spacy.Language, text: str, allowed_labels: set[str]) -> List[str]:
    ent_pairs = _extract_entities(nlp_obj, text, allowed_labels)

    # Normalize: lowercase, dedupe while preserving order
    seen = set()
    normalized: List[str] = []
    for name, _label in ent_pairs:
        if name not in seen:
            seen.add(name)
            normalized.append(name)

    return normalized

# ---------------------------
# Flask App (synchronous path)
# ---------------------------

app = Flask(__name__)

# @app.before_first_request
# def _startup():
#     _ensure_worker_started()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": SPACY_MODEL,
        "dest": {
            "host": OPENSEARCH_HOST,
            "port": OPENSEARCH_PORT,
            "ssl": OPENSEARCH_SSL,
        },
    }), 200

@app.route("/ner", methods=["POST"])
def ner():
    data = request.get_json(silent=True)
    if not data or "text" not in data or not isinstance(data["text"], str):
        return jsonify({"error": "Invalid request", "detail": "Expected JSON with a 'text' string field."}), 400

    text: str = data["text"]
    # Optional override of allowed labels
    labels_field = data.get("labels")
    if isinstance(labels_field, list) and labels_field:
        allowed = set(labels_field)
    else:
        allowed = DEFAULT_INTERESTING_ENTITY_TYPES

    entities = _extract_normalized_entities(nlp, text, allowed)

    # print with entities
    print(f"[{datetime.now(timezone.utc).isoformat()}] /ner called, {len(text)} chars, {len(entities)} entities")
    print(f"[entities discovered] {entities}\n")

    # Synchronous "heavy processing": ensure HOT index (optional) then reindex promotion
    request_id = str(uuid.uuid4())
    try:            
        status_code = 200
        payload = {
            "text": text,
            "model": SPACY_MODEL,
            "entities": entities,
            "request_id": request_id,
        }
    except requests.HTTPError as e:
        status_code = getattr(e.response, "status_code", 500) or 500
        payload = {
            "text": text,
            "model": SPACY_MODEL,
            "entities": entities,
            "error": "promotion_failed",
            "detail": str(e),
            "request_id": request_id,
            "server_response": getattr(e.response, "text", "")[:1000],
        }
    except Exception as e:
        status_code = 500
        payload = {
            "text": text,
            "model": SPACY_MODEL,
            "entities": entities,
            "error": "unexpected_error",
            "detail": f"{type(e).__name__}: {e}",
            "request_id": request_id,
        }

    return jsonify(payload), status_code

# ---------------------------
# Local Dev Entrypoint
# ---------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "8000")), debug=False)
