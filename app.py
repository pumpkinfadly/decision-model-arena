r"""Web playground for convaiinnovations/laya-multilingual.

Flask serves a single page; /api/predict runs agent.predict().
Run:  .venv\Scripts\python.exe app.py  ->  http://127.0.0.1:5001
"""
import json
import os

os.environ.setdefault("USE_TF", "0")

import threading

import laya
from flask import Flask, jsonify, render_template, request

import hier_converter

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


# Hierarchical mode config: data/contoh_pertanyaan_hierarki.json.
# Levels without "depends_on" are asked in one predict; a level with
# "depends_on" is asked in a follow-up predict using the branch criteria
# selected by the parent answer ({level_3} placeholder in instructions).
# The same shape can be supplied per-request via /api/predict_hierarchical.
HIER_PATH = os.path.join(DATA_DIR, "contoh_pertanyaan_hierarki_cabang.json")


def _norm_criteria(crit):
    return {k: v or "" for k, v in crit.items()}


def _norm_hier_config(raw):
    """Hierarchical JSON -> normalized levels dict (nulls -> '')."""
    if not isinstance(raw, dict) or not isinstance(raw.get("levels"), dict) or not raw["levels"]:
        raise ValueError(
            "config hierarki tidak valid: butuh object dengan 'levels': {qid: {type, instructions, criteria}}"
        )
    levels = {}
    for qid, q in raw["levels"].items():
        q = dict(q)
        if not str(q.get("instructions", "")).strip():
            raise ValueError(f"level '{qid}': instructions wajib")
        if isinstance(q.get("criteria"), dict):
            q["criteria"] = _norm_criteria(q["criteria"])
        elif "branches" not in q:
            raise ValueError(f"level '{qid}': butuh 'criteria' atau 'branches'")
        if isinstance(q.get("branches"), dict):
            q["branches"] = {br: _norm_criteria(c) for br, c in q["branches"].items()}
        levels[qid] = q
    return levels


def _load_hier_config():
    try:
        with open(HIER_PATH, encoding="utf-8") as f:
            return _norm_hier_config(json.load(f))
    except FileNotFoundError:
        return {}


HIER_CONFIG = _load_hier_config()


# --- converter: raw comma paths from DB -> hierarchical config -------------

@app.post("/api/convert_hierarchical")
def convert_hierarchical():
    """Raw comma paths (DB dump) -> hierarchical config JSON."""
    try:
        return jsonify(hier_converter.convert(request.get_json(force=True, silent=True)))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


def _normalize_answers(result):
    """laya predict result -> flat JSON-friendly answers dict."""
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
    return answers


def load_samples():
    """Build playground samples from files in data/.

    Each sample: (key, state file, questions file, chip label).
    v1 category options are full 4-level paths; v2 uses leaf category
    names only. Both exceed the model card's ~20-option guidance.
    """
    state_path = os.path.join(DATA_DIR, "simulasi_chat_gangguan_internet.json")
    defs = [
        (
            "contoh_1_path_koma",
            "contoh_pertanyaan_category.json",
            "Contoh 1: kategori path koma (v1)",
            None,
        ),
        (
            "contoh_2_level4",
            "contoh_pertanyaan_category_v2.json",
            "Contoh 2: kategori level 4 saja (v2)",
            None,
        ),
        (
            "contoh_3_cabang",
            None,
            "Contoh 3: semua level format cabang",
            "contoh_pertanyaan_hierarki_cabang.json",
        ),
    ]
    if not os.path.exists(state_path):
        return {}
    with open(state_path, encoding="utf-8") as f:
        data = json.load(f)
    body = "\n".join(
        f"{t.get('role', '?')}: {t.get('message', '')}".strip()
        for t in data.get("transcript", [])
    )
    samples = {}
    for key, q_file, label, hier_file in defs:
        if q_file is None:  # hierarchical sample: config from its own hier JSON
            hier_raw = {}
            try:
                with open(os.path.join(DATA_DIR, hier_file), encoding="utf-8") as f:
                    hier_raw = json.load(f)
            except OSError:
                pass
            samples[key] = {
                "label": label,
                "body": body,
                "hierarchical": True,
                "questions": hier_raw,
            }
            continue
        q_path = os.path.join(DATA_DIR, q_file)
        samples[key] = {
            "label": label,
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

    return jsonify(
        {
            "answers": _normalize_answers(result),
            "usage": result.get("usage", {}),
            "detected_language": safe_detect_language(body),
        }
    )


@app.post("/api/predict_hierarchical")
def predict_hierarchical():
    """Two-stage classification down a hierarchy.

    Levels without "depends_on" run in one predict. Each dependent level
    runs in a follow-up predict with the branch criteria chosen by its
    parent, and stage-1 answers appended to the body as context.
    Config: data/contoh_pertanyaan_hierarki.json, or per-request override
    in {"questions": <same JSON shape>}.
    """
    payload = request.get_json(force=True, silent=True) or {}
    body = str(payload.get("body", "")).strip()
    if not body:
        return jsonify({"error": "'body' harus diisi"}), 400
    try:
        cfg = (
            _norm_hier_config(payload["questions"])
            if isinstance(payload.get("questions"), dict)
            else HIER_CONFIG
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not cfg:
        return jsonify({"error": "config hierarki tidak tersedia (data/ hilang)"}), 500

    answers = {}
    input_tokens = 0

    stage1 = {qid: q for qid, q in cfg.items() if "depends_on" not in q}
    if not stage1:
        return jsonify({"error": "minimal satu level tanpa 'depends_on' diperlukan"}), 400
    try:
        with predict_lock:
            r1 = agent.predict(
                {"body": body},
                {
                    qid: {"type": q.get("type", "choice"),
                          "instructions": q["instructions"],
                          "criteria": q.get("criteria", {})}
                    for qid, q in stage1.items()
                },
            )
    except Exception as e:
        return jsonify({"error": f"predict tahap 1 gagal: {e}"}), 500
    answers.update(_normalize_answers(r1))
    input_tokens += r1.get("usage", {}).get("input_tokens", 0)

    for qid, q in cfg.items():
        dep = q.get("depends_on")
        if not dep:
            continue
        parent = answers.get(dep, {}).get("choice")
        crit = q.get("branches", {}).get(parent) if parent else None
        if not crit:
            continue  # branch tanpa kriteria: level dilewati
        instr = q["instructions"]
        for k, a in answers.items():
            if a.get("choice"):
                instr = instr.replace("{" + k + "}", a["choice"])
        context = body + "\n\nKonteks klasifikasi sebelumnya: " + " > ".join(
            a["choice"] for a in answers.values() if a.get("choice")
        )
        try:
            with predict_lock:
                r2 = agent.predict(
                    {"body": context},
                    {qid: {"type": q.get("type", "choice"),
                           "instructions": instr,
                           "criteria": crit}},
                )
        except Exception as e:
            return jsonify({"error": f"predict tahap 2 ({qid}) gagal: {e}"}), 500
        answers[qid] = _normalize_answers(r2)[qid]
        input_tokens += r2.get("usage", {}).get("input_tokens", 0)

    path_parts = [a.get("choice") for a in answers.values() if a.get("choice")]
    return jsonify(
        {
            "answers": answers,
            "final_path": " > ".join(path_parts),
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
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
