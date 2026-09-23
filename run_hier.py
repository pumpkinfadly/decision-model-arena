"""Test /api/predict_hierarchical with an ISP sample (default: contoh_3_cabang).

Usage: python run_hier.py [sample_key]
"""
import json
import sys
import urllib.request

key = sys.argv[1] if len(sys.argv) > 1 else "contoh_3_cabang"
s = json.load(urllib.request.urlopen("http://127.0.0.1:5001/api/samples", timeout=15))[key]
print("label:", s["label"])

payload = {"body": s["body"]}
if isinstance(s.get("questions"), dict) and s["questions"].get("levels"):
    payload["questions"] = s["questions"]  # per-sample hierarchical config

req = urllib.request.Request(
    "http://127.0.0.1:5001/api/predict_hierarchical",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
r = json.load(urllib.request.urlopen(req, timeout=300))
print("FINAL PATH:", r["final_path"])
for lv in ("level_1", "level_2", "level_3", "level_4"):
    a = r["answers"].get(lv)
    if not a:
        continue
    top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:3]
    print(f"{lv}: {a['choice']}  conf={a['confidence']:.3f}  top3={[(k, round(p, 3)) for k, p in top]}")
print("usage:", r["usage"])
