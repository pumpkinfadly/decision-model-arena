r"""Shared two-stage hierarchical flow + config normalizer.

Pure Python, no heavy deps — imported by both the laya server (5001) and the
Julia server (5002) so the two models run the identical flow.

run_hierarchical(predict_fn, body, cfg):
  Stage 1: levels without "depends_on" in one predict call.
  Each dependent level: follow-up predict with the branch criteria chosen by
  its parent answer; stage-so-far answers appended to the body as context.

predict_fn(body, questions_dict) -> normalized answers dict:
  {qid: {type, choice|yes|score, probabilities, confidence}}
"""


def norm_criteria(crit):
    return {k: v or "" for k, v in crit.items()}


def norm_hier_config(raw):
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
            q["criteria"] = norm_criteria(q["criteria"])
        elif "branches" not in q:
            raise ValueError(f"level '{qid}': butuh 'criteria' atau 'branches'")
        if isinstance(q.get("branches"), dict):
            q["branches"] = {br: norm_criteria(c) for br, c in q["branches"].items()}
        levels[qid] = q
    return levels


def hier_questions(cfg, only_ids=None):
    """Levels -> plain predict questions {qid: {type, instructions, criteria}}."""
    out = {}
    for qid, q in cfg.items():
        if only_ids is not None and qid not in only_ids:
            continue
        out[qid] = {
            "type": q.get("type", "choice"),
            "instructions": q["instructions"],
            "criteria": q.get("criteria", {}),
        }
    return out


def run_hierarchical(predict_fn, body, cfg):
    """Run the staged flow; returns (answers, final_path)."""
    stage1_ids = [qid for qid, q in cfg.items() if "depends_on" not in q]
    if not stage1_ids:
        raise ValueError("minimal satu level tanpa 'depends_on' diperlukan")
    answers = predict_fn(body, hier_questions(cfg, set(stage1_ids)))

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
        got = predict_fn(context, {qid: {"type": q.get("type", "choice"), "instructions": instr, "criteria": crit}})
        answers[qid] = got[qid]

    final_path = " > ".join(a["choice"] for a in answers.values() if a.get("choice"))
    return answers, final_path
