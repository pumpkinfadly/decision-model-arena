r"""laya-multilingual server (port 5001) — playground + compare hub.

Loads convaiinnovations/laya-multilingual once. Serves the playground UI,
the /compare UI, and fans /api/compare out to this model and the Julia-1
server (julia_server.py, port 5002).
Run:  .venv\Scripts\python.exe laya_server.py  ->  http://127.0.0.1:5001
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request

os.environ.setdefault("USE_TF", "0")

import laya
from flask import Flask, jsonify, render_template, request

import hier_converter
from hier_flow import norm_hier_config, run_hierarchical

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JULIA_URL = "http://127.0.0.1:5002"

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


# Hierarchical mode config: data/contoh_pertanyaan_hierarki_cabang.json.
# Flow logic lives in hier_flow.py (shared with julia_server.py); config
# normalizer too. The same shape can be supplied per-request via
# /api/predict_hierarchical.
HIER_PATH = os.path.join(DATA_DIR, "contoh_pertanyaan_hierarki_cabang.json")


def _load_hier_config():
    try:
        with open(HIER_PATH, encoding="utf-8") as f:
            return norm_hier_config(json.load(f))
    except FileNotFoundError:
        return {}


HIER_CONFIG = _load_hier_config()


def _laya_predict_fn(body, questions):
    """predict_fn contract for hier_flow."""
    with predict_lock:
        result = agent.predict({"body": body}, questions)
    return _normalize_answers(result)


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
            norm_hier_config(payload["questions"])
            if isinstance(payload.get("questions"), dict)
            else HIER_CONFIG
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not cfg:
        return jsonify({"error": "config hierarki tidak tersedia (data/ hilang)"}), 500

    try:
        answers, final_path = run_hierarchical(_laya_predict_fn, body, cfg)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"predict hierarki gagal: {e}"}), 500
    return jsonify(
        {
            "answers": answers,
            "final_path": final_path,
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


# --- compare: laya (local) vs Julia-1 (proxy to julia_server:5002) ---------

@app.get("/compare")
def compare_page():
    return render_template("compare.html")


def _julia_call(path, payload, timeout=300):
    req = urllib.request.Request(
        JULIA_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


@app.post("/api/compare")
def api_compare():
    payload = request.get_json(force=True, silent=True) or {}
    body = str(payload.get("body", "")).strip()
    if not body:
        return jsonify({"error": "'body' harus diisi"}), 400
    hier = bool(payload.get("hierarchical"))
    questions = payload.get("questions")

    cfg = None
    if hier:
        try:
            cfg = norm_hier_config(questions) if isinstance(questions, dict) else HIER_CONFIG
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if not cfg:
            return jsonify({"error": "config hierarki tidak tersedia"}), 400

    results = {}

    def run_laya():
        t0 = time.perf_counter()
        try:
            if hier:
                answers, final_path = run_hierarchical(_laya_predict_fn, body, cfg)
                results["laya"] = {
                    "answers": answers,
                    "final_path": final_path,
                    "latency_ms": round((time.perf_counter() - t0) * 1000),
                }
            else:
                b, qs = build_questions({"body": body, "questions": questions})
                with predict_lock:
                    raw = agent.predict({"body": b}, qs)
                results["laya"] = {
                    "answers": _normalize_answers(raw),
                    "usage": raw.get("usage", {}),
                    "latency_ms": round((time.perf_counter() - t0) * 1000),
                }
        except ValueError as e:
            results["laya"] = {"error": str(e)}
        except Exception as e:
            results["laya"] = {"error": f"laya gagal: {e}"}

    def run_julia():
        try:
            if hier:
                jp = {"body": body}
                if isinstance(questions, dict):
                    jp["questions"] = questions
                results["julia"] = _julia_call("/predict_hierarchical", jp)
            else:
                results["julia"] = _julia_call(
                    "/predict", {"body": body, "questions": questions}
                )
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode()).get("error", "")
            except Exception:
                detail = ""
            results["julia"] = {"error": detail or f"julia HTTP {e.code}"}
        except Exception as e:
            results["julia"] = {"error": f"julia server tidak terjangkau: {e}"}

    threads = [threading.Thread(target=run_laya), threading.Thread(target=run_julia)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return jsonify({"laya": results.get("laya"), "julia": results.get("julia")})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, threaded=True)
