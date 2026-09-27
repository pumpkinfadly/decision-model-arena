r"""Julia-1 comparison server (port 5002).

Runs in its own venv (.venv-julia: transformers<5.1) isolated from the laya
server. Same request/response shapes as the laya endpoints so /api/compare
can fan out to both. Choice questions with >20 options fall back to the
Julia Router (winner + confidence only, conditional distribution).
"""
import json
import os
import threading
import time

os.environ.setdefault("JULIA_CPU_THREADS", "4")

from flask import Flask, jsonify, request
from hier_flow import norm_hier_config, run_hierarchical
from julia import load_model

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HIER_PATH = os.path.join(DATA_DIR, "contoh_pertanyaan_hierarki_cabang.json")

app = Flask(__name__)
engine = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "julia_model"))
lock = threading.Lock()

QTYPES = {"choice", "score", "noul"}
MAX_NATIVE_OPTIONS = 20

try:
    from julia.router import Router

    router = Router(engine, survivors=2)
except Exception as e:  # native bend backend may be unavailable
    print(f"Warning: Julia Router unavailable ({e}); >20-option questions will error")
    router = None


def adapt_questions(questions):
    """laya-format questions -> julia format (same keys; desc null -> label)."""
    qs = {}
    for qid, q in questions.items():
        qtype = q.get("type")
        crit = q.get("criteria")
        if qtype == "choice" and isinstance(crit, dict):
            crit = {k: (v or k) for k, v in crit.items()}
        if qtype == "noul":
            crit = None
        qs[qid] = {"instructions": q.get("instructions", ""), "type": qtype, "criteria": crit}
    return qs


def norm_answers(raw):
    """Julia typed answers -> laya-normalized shape."""
    out = {}
    for qid, a in raw.items():
        r = {"type": a.get("type"), "probabilities": a.get("probabilities", {})}
        if r["type"] == "choice":
            r["choice"] = a.get("choice")
            r["confidence"] = a.get("max_probability")
        elif r["type"] == "noul":
            r["yes"] = a.get("noul")
            p = a.get("probabilities", {})
            r["confidence"] = max(p.get("true", 0.0), p.get("false", 0.0))
        elif r["type"] == "score":
            r["score"] = a.get("score")
            r["confidence"] = a.get("max_probability")
        out[qid] = r
    return out


def julia_predict_fn(body, questions):
    """predict_fn contract for hier_flow; native typed only (<=20 options)."""
    with lock:
        raw = engine.predict(state=body, questions=adapt_questions(questions))
    return norm_answers(raw["answers"])


def validate_flat(payload):
    if not isinstance(payload, dict) or not str(payload.get("body", "")).strip():
        raise ValueError("'body' harus diisi")
    raw = payload.get("questions", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError("minimal satu pertanyaan diperlukan")
    questions = {}
    for i, q in enumerate(raw):
        qid = str(q.get("id", "")).strip()
        qtype = str(q.get("type", "")).strip()
        instr = str(q.get("instructions", "")).strip()
        if not qid or qid in questions:
            raise ValueError(f"pertanyaan {i + 1}: nama unik wajib")
        if qtype not in QTYPES or not instr:
            raise ValueError(f"pertanyaan '{qid}': type/instructions tidak valid")
        if qtype == "choice":
            crit = q.get("criteria") or {}
            questions[qid] = {"type": "choice", "instructions": instr, "criteria": dict(crit)}
        elif qtype == "score":
            levels = [str(v) for v in (q.get("criteria") or [])]
            questions[qid] = {"type": "score", "instructions": instr, "criteria": levels}
        else:
            questions[qid] = {"type": "noul", "instructions": instr}
    return payload["body"].strip(), questions


def route_big_choice(body, instructions, labels):
    """Options >20: winner + confidence.

    Prefers the native Julia Router; if its bend runtime is unavailable
    (e.g. Windows without the native build), falls back to a chunked
    two-stage: top-2 per group of 20, then a final predict of the survivors.
    """
    labels = list(labels)
    if router is not None:
        with lock:
            result = router.route({"state": body, "question": instructions, "options": labels})
        if isinstance(result, dict):
            return {"mode": "router", "choice": result.get("choice"), "confidence": result.get("confidence")}
        return {"mode": "router", "choice": getattr(result, "choice", None), "confidence": getattr(result, "confidence", None)}

    groups = [labels[i:i + MAX_NATIVE_OPTIONS] for i in range(0, len(labels), MAX_NATIVE_OPTIONS)]
    survivors = []
    for g in groups:
        ans = julia_predict_fn(body, {
            "_chunk": {"type": "choice", "instructions": instructions, "criteria": {k: k for k in g}}
        })["_chunk"]
        survivors.extend(k for k, _ in sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])[:2])
    final = julia_predict_fn(body, {
        "_final": {"type": "choice", "instructions": instructions, "criteria": {k: k for k in survivors}}
    })["_final"]
    return {
        "mode": "chunked",
        "choice": final.get("choice"),
        "confidence": final.get("confidence"),
        # probabilities of the final survivors round (conditional, not a
        # global distribution over all options) — enough for top-3 display
        "probabilities": final.get("probabilities"),
    }


@app.get("/health")
def health():
    return jsonify({"ok": True, "router": router is not None})


@app.post("/predict")
def predict():
    try:
        body, questions = validate_flat(request.get_json(force=True, silent=True))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    t0 = time.perf_counter()
    small = {k: v for k, v in questions.items()
             if not (v["type"] == "choice" and len(v.get("criteria", {})) > MAX_NATIVE_OPTIONS)}
    big = {k: v for k, v in questions.items()
           if v["type"] == "choice" and len(v.get("criteria", {})) > MAX_NATIVE_OPTIONS}
    answers = {}
    try:
        if small:
            answers.update(julia_predict_fn(body, small))
        for qid, q in big.items():
            r = route_big_choice(body, q["instructions"], q["criteria"].keys())
            answers[qid] = {
                "type": "choice",
                "mode": r["mode"],
                "choice": r["choice"],
                "confidence": r["confidence"],
                "probabilities": r.get("probabilities", {}),
            }
    except Exception as e:
        return jsonify({"error": f"julia predict gagal: {e}"}), 500
    return jsonify({"answers": answers, "latency_ms": round((time.perf_counter() - t0) * 1000)})


@app.post("/predict_hierarchical")
def predict_hierarchical():
    payload = request.get_json(force=True, silent=True) or {}
    body = str(payload.get("body", "")).strip()
    if not body:
        return jsonify({"error": "'body' harus diisi"}), 400
    try:
        if isinstance(payload.get("questions"), dict):
            cfg = norm_hier_config(payload["questions"])
        else:
            with open(HIER_PATH, encoding="utf-8") as f:
                cfg = norm_hier_config(json.load(f))
    except (ValueError, OSError) as e:
        return jsonify({"error": str(e)}), 400

    t0 = time.perf_counter()
    try:
        answers, final_path = run_hierarchical(julia_predict_fn, body, cfg)
    except Exception as e:
        return jsonify({"error": f"julia hierarchical gagal: {e}"}), 500
    return jsonify({
        "answers": answers,
        "final_path": final_path,
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, threaded=True)
