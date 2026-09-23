"""Run the ISP sample (transcript state + single category question) via API."""
import json
import urllib.request

s = json.load(urllib.request.urlopen("http://127.0.0.1:5001/api/samples", timeout=15))[
    "gangguan_internet_id"
]
print("questions:", list(s["questions"]))

req = urllib.request.Request(
    "http://127.0.0.1:5001/api/predict",
    data=json.dumps(
        {
            "body": s["body"],
            "questions": [dict(id=k, **v) for k, v in s["questions"].items()],
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
)
r = json.load(urllib.request.urlopen(req, timeout=180))
a = r["answers"]["category"]
print("CHOICE:", a["choice"])
print("CONFIDENCE:", a["confidence"])
for k, p in sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:8]:
    print(f"{p:.4f}  {k}")
print("usage:", r["usage"])
