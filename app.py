"""Web playground for convaiinnovations/laya-multilingual.

Flask serves a single page; /api/predict runs agent.predict().
Run:  .venv\Scripts\python.exe app.py  ->  http://127.0.0.1:5001
"""
import json
import os

os.environ.setdefault("USE_TF", "0")

import threading

import laya
from flask import Flask, jsonify, render_template, request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

app = Flask(__name__)
agent = laya.load("convaiinnovations/laya-multilingual")
predict_lock = threading.Lock()  # torch model, single predict at a time

QTYPES = {"choice", "score", "noul"}

PRESETS = {
    "triage": laya.triage_questions,
    "email": laya.email_questions,
    "router": laya.router_questions,
    "moderation": laya.moderation_questions,
}

# Sample state: data/simulasi_chat_gangguan_internet.json (transcript).
# Sample questions: data/contoh_pertanyaan_category.json — single user-supplied
# `category` choice question, 34 full-path options (null descriptions -> "").
# 34 options exceeds the model card's ~20-option guidance — kept as provided.


def _load_question_file(path):
    """Question JSON -> laya questions dict; null criteria values -> ''."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    questions = {}
    for qid, q in raw.items():
        q = dict(q)
        if isinstance(q.get("criteria"), dict):
            q["criteria"] = {k: v or "" for k, v in q["criteria"].items()}
        questions[qid] = q
    return questions


def load_samples():
    """Build playground samples from files in data/."""
    samples = {}
    state_path = os.path.join(DATA_DIR, "simulasi_chat_gangguan_internet.json")
    q_path = os.path.join(DATA_DIR, "contoh_pertanyaan_category.json")
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            data = json.load(f)
        body = "\n".join(
            f"{t.get('role', '?')}: {t.get('message', '')}".strip()
            for t in data.get("transcript", [])
        )
        samples["gangguan_internet_id"] = {
            "label": "Contoh: ISP \u00b7 gangguan internet (ID)",
            "body": body,
            "questions": _load_question_file(q_path) if os.path.exists(q_path) else {},
        }
    return samples


SAMPLES = load_samples()


def build_questions(payload):
    """Validate frontend question list -> laya questions dict."""
    if not isinstance(payload, dict) or "body" not in payload:
        raise ValueError("JSON body with 'body' field required")
    body = str(payload["body"]).strip()
    if not body:
        raise ValueError("'body' must not be empty")
    raw = payload.get("questions", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError("at least one question required")

    questions = {}
    for i, q in enumerate(raw):
        qid = str(q.get("id", "")).strip()
        qtype = str(q.get("type", "")).strip()
        instr = str(q.get("instructions", "")).strip()
        if not qid:
            raise ValueError(f"question {i + 1}: name required")
        if qid in questions:
            raise ValueError(f"duplicate question name: {qid}")
        if qtype not in QTYPES:
            raise ValueError(f"question '{qid}': type must be one of {sorted(QTYPES)}")
        if not instr:
            raise ValueError(f"question '{qid}': instructions required")

        if qtype == "choice":
            criteria = q.get("criteria") or {}
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"question '{qid}': choice needs >= 2 options")
            criteria = {str(k).strip(): str(v).strip() for k, v in criteria.items()}
            if any(not k for k in criteria):
                raise ValueError(f"question '{qid}': option labels required")
            questions[qid] = {
                "type": "choice",
                "instructions": instr,
                "criteria": criteria,
            }
        elif qtype == "score":
            levels = q.get("criteria") or []
            if not isinstance(levels, list) or len(levels) < 2:
                raise ValueError(f"question '{qid}': score needs >= 2 ordered levels")
            levels = [str(v).strip() for v in levels]
            if any(not v for v in levels):
                raise ValueError(f"question '{qid}': empty score level")
            questions[qid] = {
                "type": "score",
                "instructions": instr,
                "criteria": levels,
            }
        else:  # noul
            questions[qid] = {"type": "noul", "instructions": instr}
    return body, questions


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/presets")
def presets():
    return jsonify(
        {
            name: fn()
            for name, fn in PRESETS.items()
        }
    )


@app.get("/api/samples")
def samples():
    return jsonify(SAMPLES)


@app.post("/api/predict")
def predict():
    try:
        body, questions = build_questions(request.get_json(force=True, silent=True))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        with predict_lock:
            result = agent.predict({"body": body}, questions)
    except Exception as e:  # surface backend failures to UI
        return jsonify({"error": f"predict failed: {e}"}), 500

    answers = {}
    for qid, ans in result.get("answers", {}).items():
        out = {"type": ans.get("type")}
        if out["type"] == "choice":
            out["choice"] = ans.get("choice")
            out["probabilities"] = ans.get("probabilities", {})
        elif out["type"] == "noul":
            out["yes"] = ans.get("noul")
        elif out["type"] == "score":
            out["score"] = ans.get("score")
            out["legend"] = ans.get("legend", {})
            out["probabilities"] = ans.get("probabilities", {})
        out["confidence"] = ans.get("confidence")
        out["act_probability"] = ans.get("action", {}).get("act_probability")
        answers[qid] = out

    return jsonify(
        {
            "answers": answers,
            "usage": result.get("usage", {}),
            "detected_language": safe_detect_language(body),
        }
    )


def safe_detect_language(text):
    try:
        r = laya.detect_language(text)
        if isinstance(r, dict):
            return str(r.get("language", "?"))
        return str(r or "?")
    except Exception:
        return "?"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, threaded=True)
