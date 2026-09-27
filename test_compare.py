"""Test /api/compare: hierarchical (contoh 3) and flat (contoh 1)."""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:5001"
samples = json.load(urllib.request.urlopen(f"{BASE}/api/samples", timeout=15))


def compare(payload):
    req = urllib.request.Request(
        BASE + "/api/compare", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=600))


key = sys.argv[1] if len(sys.argv) > 1 else "contoh_3_cabang"
s = samples[key]
hier = bool(s.get("hierarchical"))
payload = {"body": s["body"], "hierarchical": hier}
if hier:
    payload["questions"] = s["questions"]
else:
    payload["questions"] = [dict(id=k, **v) for k, v in s["questions"].items()]

r = compare(payload)
for side in ("laya", "julia"):
    d = r.get(side) or {}
    if "error" in d:
        print(f"{side}: ERROR {d['error']}")
        continue
    line = f"{side}: {d.get('latency_ms')} ms"
    if d.get("final_path"):
        line += f" | {d['final_path']}"
    print(line)
    for qid, a in (d.get("answers") or {}).items():
        val = a.get("choice") or a.get("yes") or a.get("score")
        extra = f" mode={a['mode']}" if a.get("mode") else ""
        print(f"  {qid}: {val} conf={a.get('confidence')}{extra}")
