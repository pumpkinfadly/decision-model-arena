"""Benchmark Contoh 1/2/3: latency + tokens (3 runs each)."""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:5001"
samples = json.load(urllib.request.urlopen(f"{BASE}/api/samples", timeout=15))


def call(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return (time.perf_counter() - t0) * 1000, r


CASES = [
    ("Contoh 1 (flat 34 path)", "contoh_1_path_koma", "/api/predict"),
    ("Contoh 2 (flat 34 leaf)", "contoh_2_level4", "/api/predict"),
    ("Contoh 3 (bertingkat cabang)", "contoh_3_cabang", "/api/predict_hierarchical"),
]

for label, key, path in CASES:
    s = samples[key]
    payload = {"body": s["body"]}
    if path.endswith("hierarchical") and s.get("questions", {}).get("levels"):
        payload["questions"] = s["questions"]
    else:
        payload["questions"] = [dict(id=k, **v) for k, v in s["questions"].items()]
    times = []
    for _ in range(3):
        ms, r = call(path, payload)
        times.append(ms)
    tok = r.get("usage", {}).get("input_tokens", 0)
    n_predict = 4 if "hierarchical" in path else 1
    print(f"{label}: avg {sum(times)/len(times):.0f} ms (runs: {[round(t) for t in times]}), "
          f"tokens {tok}, predicts {n_predict}")
