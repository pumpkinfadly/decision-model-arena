# Alur Backend — decision-model arena

Dokumen ini menjelaskan arsitektur backend workspace versus model keputusan.
Saat ini berisi dua model: **laya-multilingual** (port 5001) dan **Julia-1**
(port 5002). Semua logika bersama ada di modul tanpa dependensi berat agar
bisa dipakai kedua server.

## Gambaran arsitektur

```
browser
  │
  ├── /                     playground single-model (laya)
  ├── /compare              arena versus (2 model)
  │
  ▼
laya_server.py (5001)                    julia_server.py (5002)
  ├─ /api/predict           (laya)         ├─ /predict        (julia)
  ├─ /api/predict_hierarchical             ├─ /predict_hierarchical
  ├─ /api/convert_hierarchical             └─ /health
  ├─ /api/presets /api/samples
  └─ /api/compare ──────── fan-out paralel (thread per model)
        │                                   ▲
        └── HTTP proxy ─────────────────────┘
```

Alasan dua proses: Julia-1 mematok `transformers>=5.0,<5.1` sedangkan laya
jalan di 5.17 — venv terpisah (`.venv` dan `.venv-julia`), server terpisah,
komunikasi via HTTP lokal. RAM: laya ~1,7 GB + Julia ~1 GB.

## Modul bersama

- `hier_flow.py` — alur klasifikasi bertingkat model-agnostic:
  `run_hierarchical(predict_fn, body, cfg)`. Level tanpa `depends_on`
  ditanya dalam satu predict; setiap level dependent ditanya pada predict
  lanjutan dengan opsi cabang hasil induknya, dan jawaban tahap sebelumnya
  disuntikkan ke body sebagai konteks. Juga berisi `norm_hier_config`
  (validasi + null → `""`).
- `hier_converter.py` — konverter path koma dari DB menjadi config
  hierarki; dipakai endpoint `/api/convert_hierarchical`, CLI
  `tools/convert_from_db.py`, dan auto-convert saat upload di UI.

Kedua server memanggil `run_hierarchical` dengan `predict_fn` masing-masing,
sehingga kedua model menjalankan alur yang persis sama — prasyarat adil
untuk perbandingan.

## Kontrak predict_fn

```python
def predict_fn(body: str, questions: dict) -> dict[str, dict]:
    """questions: {qid: {type, instructions, criteria}}
    return: {qid: {type, choice|yes|score, probabilities, confidence}}
    """
```

- **laya** (`_laya_predict_fn`): bungkus `agent.predict({"body": ...}, qs)`
  + `_normalize_answers` → menambah `act_probability` dari act/escalate head.
- **Julia** (`julia_predict_fn`): `engine.predict(state=body, questions=...)`
  dengan adaptasi kecil: criteria `choice` bernilai null diganti label opsi
  itu sendiri (Julia merender deskripsi sebagai teks opsi; string kosong
  merusak skor), `noul` selalu `criteria=None`. Normalisasi keluaran:
  `max_probability` → `confidence`, `noul` → `yes`.

## Alur `/api/compare`

```
POST {body, hierarchical?, questions}
  │
  ├─ thread A: laya
  │    hierarchical?  run_hierarchical(_laya_predict_fn, ...)
  │    else:          build_questions + agent.predict (flat)
  │
  ├─ thread B: julia (HTTP ke 5002)
  │    hierarchical?  /predict_hierarchical
  │    else:          /predict
  │
  ▼
{laya: {answers, final_path?, latency_ms} | {error},
 julia: {answers, final_path?, latency_ms} | {error}}
```

Kedua sisi berjalan paralel; kegagalan satu sisi tidak membatalkan sisi
lain — frontend menampilkan kartu error untuk sisi yang gagal.

## Alur bertingkat (dijalankan identik oleh kedua model)

```
body (transcript)
  │ validasi + norm_hier_config
  ▼
TAHAP 1 — satu predict untuk semua level tanpa depends_on
  (level_1 3 opsi · level_2 5 opsi · level_3 11 opsi)
  ▼
TAHAP 2 — per level dependent, sekuensial
  parent = jawaban level induk
  opsi   = branches[parent]          ← subset kecil (≤ 7)
  body   = body + "\n\nKonteks klasifikasi sebelumnya: <jawaban-jawaban>"
  instruksi: placeholder {level_N} diganti jawaban terkait
  ▼
final_path = " > ".join(jawaban semua level)
```

## Opsi lebih dari 20 (Julia)

Julia native mendukung 2–20 opsi per `choice`. Router resminya butuh runtime
bend native (`libjulia_router.so`) yang tidak ter-build di Windows, jadi
`julia_server.py` memakai **fallback chunked** sendiri: pecah opsi jadi grup
20 → top-2 tiap grup → satu predict final antar survivor. Hasil: winner +
confidence + probabilitas ronde final (kondisional atas survivor, bukan
distribusi global 34 opsi). Mode ditandai `mode: "chunked"` di respons dan
tag di UI.

## Contoh hasil nyata (transcript ISP)

| level | laya | Julia-1 |
|---|---|---|
| level_1 | Komplain 97% | Komplain 66% |
| level_2 | Internet 98% | Internet 86% |
| level_3 | Gangguan Internet 81% | Gangguan Internet 98% |
| level_4 | Putus-Putus 53% | **LOS Merah 97%** (ground truth taksonomi) |

Flat-34: kedua model salah memilih leaf; conf 4,6% (laya) vs 45,9% (Julia,
chunked). Kesimpulan desain tetap: pecah per level / per cabang, jangan
satu pertanyaan besar.

## Menambah model baru

Lima langkah (detail di README): venv sendiri → `<model>_server.py` di port
baru dengan endpoint bentuk sama → adapter `predict_fn` + `hier_flow` →
tambah thread fan-out di `/api/compare` → kolom baru di `compare.html`.
