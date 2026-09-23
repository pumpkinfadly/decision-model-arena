# laya-multilingual (local)

convaiinnovations/laya-multilingual — 322M non-autoregressive decision/routing model.
Given state (text/email/ticket/JSON) + typed questions, returns typed answers with
probabilities in one forward pass. No text generation. Apache 2.0.

## Activate

```cmd
.venv\Scripts\activate
```

Python 3.13, laya 0.3.7, torch 2.14.0 (CPU). Weights cached under
`%USERPROFILE%\.cache\huggingface\hub\` after first download.

## Run

```cmd
set USE_TF=0 && python test_laya.py
```

## Web playground

```cmd
.venv\Scripts\python.exe app.py
```

→ http://127.0.0.1:5001

- Input state textarea + example texts (ID + EN only), language detection shown after run
- Upload JSON: state file (`{"transcript": [...]}` → joined as body, or `{"body": "..."}`) + questions file (`{qid: {type, instructions, criteria}}`, null criteria → "")
- Question builder: `choice` (options), `score` (ordered levels), `noul` (yes/no) — or load presets (triage / email / router / moderation)
- Results: per-question probability bars, confidence, low-confidence escalate warning, token usage
- API: `POST /api/predict` (JSON), `GET /api/presets`, `GET /api/samples`

## Sample data (ID)

- `data/simulasi_chat_gangguan_internet.json` — state: simulated Indonesian ISP
  chat (transcript + taxonomy). Uploadable as state file.
- `data/contoh_pertanyaan_category.json` — questions: single `category` choice
  question, 34 full-path options (Level1 > ... > Level4). Uploadable as
  questions file.
- Chip "Contoh: ISP · gangguan internet (ID)" loads both.
- Zero-shot result: near-chance (top pick 8.3%, confidence 4.1%) — 34 options
  exceeds model card's ~20-option limit + most option descriptions are null.
  Split-per-level question set (7 questions) routed correctly instead.

## Usage

```python
import laya
agent = laya.load("convaiinnovations/laya-multilingual")
result = agent.predict(state_dict, questions_dict)  # see test_laya.py
```

Question types: `choice`, `noul` (yes/no/unknown/unsure), others per laya docs.

## Notes from smoke test (2026-09-23)

- English duplicate-charge email → `billing`, confidence 1.0. Correct.
- Same email in Hindi → `technical` 0.67, confidence 0.26. Wrong, but low
  confidence — matches model card: uncalibrated, zero-shot typed decisions
  near-chance. Check `confidence` / escalate rather than trusting choice blindly.
- Context limit 1024 tokens per question. Keep choice questions < ~20 options.
