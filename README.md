# decision-model arena

Workspace untuk **versus model keputusan (decision/routing model)** — model
non-generatif yang mengubah *state + pertanyaan terstruktur* menjadi jawaban
berprobabilitas dalam satu forward pass. Saat ini berisi dua model yang
dibandingkan head-to-head:

| | laya-multilingual | Julia-1 |
|---|---|---|
| pembuat | convaiinnovations | Supersonic Labs |
| ukuran | 322M (mmBERT-base) | 144M (mmBERT-small) |
| tipe pertanyaan | choice / score / noul | choice / score / noul (format identik) |
| RAM proses | ~1,7 GB | ~1 GB |
| lisensi | Apache-2.0 | Apache-2.0 |
| catatan | act/escalate head, temperature, PyPI | id-ID 81,2% (MASSIVE), ONNX WebGPU |

Hasil pengukuran di transcript contoh ada di bagian [Temuan](#temuan).

## Struktur

```
laya_server.py          server laya (5001): playground + /compare hub
julia_server.py         server Julia-1 (5002): venv sendiri (transformers<5.1)
hier_flow.py            alur klasifikasi bertingkat — shared, model-agnostic
hier_converter.py       konverter path koma (DB) -> config hierarki
templates/index.html    playground (single model)
templates/compare.html  arena versus (2 model paralel)
data/                   state + pertanyaan contoh (format-neutral)
tools/convert_from_db.py CLI konverter DB dump
.venv/ .venv-julia/     venv per model (versi dependensi bisa bentrok)
julia_model/            snapshot Julia-1 (577 MB, git-ignored)
```

## Menjalankan

```cmd
:: server 1 — laya (5001): playground + halaman compare
.venv\Scripts\python.exe laya_server.py

:: server 2 — Julia-1 (5002), venv terpisah
.venv-julia\Scripts\python.exe julia_server.py
```

- Playground single-model: http://127.0.0.1:5001
- Arena versus: http://127.0.0.1:5001/compare — butuh dua server hidup;
  satu mati → sisi itu jadi kartu error, sisi lain tetap jalan

Setup Julia-1 (sekali, sudah dilakukan):

```cmd
.venv\Scripts\python.exe setup_julia.py          :: snapshot -> julia_model/
python -m venv .venv-julia
.venv-julia\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-julia\Scripts\pip install "transformers>=5.0,<5.1" safetensors numpy huggingface_hub flask
.venv-julia\Scripts\pip install -e julia_model --no-deps
```

## API

| Endpoint | Server | Fungsi |
|---|---|---|
| `POST /api/predict` | 5001 | predict laya (flat) |
| `POST /api/predict_hierarchical` | 5001 | bertingkat laya |
| `POST /api/convert_hierarchical` | 5001 | path koma -> config hierarki |
| `GET /api/presets` `GET /api/samples` | 5001 | preset laya + contoh |
| `POST /predict` `POST /predict_hierarchical` | 5002 | padanan Julia (bentuk sama) |
| `GET /health` | 5002 | status Julia + Router |
| `POST /api/compare` | 5001 | **fan-out paralel dua model** |

Format pertanyaan identik untuk kedua model:
`{qid: {type, instructions, criteria}}` (choice: dict label→deskripsi,
score: list berurut, noul: tanpa criteria). Config bertingkat:
`{levels: {qid: {criteria | branches + depends_on}}}`.

## Menambah model ketiga (checklist)

1. Venv sendiri bila dependensi bentrok: `.venv-<model>`
2. Server baru `<model>_server.py` di port berikutnya — expose
   `POST /predict` + `POST /predict_hierarchical` dengan bentuk respons sama
   (`answers` ternormalisasi + `latency_ms`)
3. Adapter: fungsi `<model>_predict_fn(body, questions)` mengembalikan
   `{qid: {type, choice|yes|score, probabilities, confidence}}`, lalu
   panggil `hier_flow.run_hierarchical` untuk mode bertingkat
4. Di `laya_server.py`: tambah thread fan-out di `/api/compare`
5. UI: kolom baru di `templates/compare.html` (kelas `.mcard.<model>`)

## Data contoh

- `data/simulasi_chat_gangguan_internet.json` — state: transcript ISP
  Indonesia + taksonomi kategori 4 level
- `data/contoh_pertanyaan_category.json` — Contoh 1: opsi path koma (34)
- `data/contoh_pertanyaan_category_v2.json` — Contoh 2: opsi level-4 saja
- `data/contoh_pertanyaan_hierarki_cabang.json` — Contoh 3: bertingkat
  penuh (depends_on + branches di semua level)

Upload UI menerima ketiga format + list path koma mentah (auto-convert).

## Temuan (transcript ISP, 2026-09)

| desain | laya | Julia-1 |
|---|---|---|
| Contoh 1 flat-34 | salah, conf 4,6% | salah (chunked), conf 45,9% |
| Contoh 3 bertingkat | benar path, leaf `Putus-Putus` 53% | **benar + leaf ground truth `LOS Merah` 97%** |
| latensi bertingkat | ~1.750 ms | ~1.100 ms |

Pola: opsi ≤ 11 per pertanyaan + struktur bertingkat = akurasi tinggi di
kedua model; flat-34 lemah di keduanya. Julia Router native tidak ter-build
di Windows → fallback chunked (top-2 per grup 20, final survivor;
probabilitas kondisional, bukan distribusi global).

Detail alur backend: [BACKEND_FLOW.md](BACKEND_FLOW.md).

## Deploy server (Ubuntu/Debian)

Kebutuhan per model: laya 2 vCPU/4 GB; Julia ~1 GB → keduanya + OS muat di
4 GB. Swap bila RAM terbatas (proses laya sendiri ~1,7 GB, tanpa opsi
half/quant di laya 0.3.7).

```bash
sudo apt update && sudo apt install -y python3-venv git
git clone https://github.com/pumpkinfadly/laya-multilingual-playground.git
cd laya-multilingual-playground

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install gunicorn
USE_TF=0 python -c "import laya; laya.load('convaiinnovations/laya-multilingual')"
USE_TF=0 gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:5001 laya_server:app

# Julia (venv kedua) — gunicorn 1 worker
python3 -m venv .venv-julia && source .venv-julia/bin/activate
python setup_julia.py
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=5.0,<5.1" safetensors numpy huggingface_hub flask gunicorn
pip install -e julia_model --no-deps
gunicorn -w 1 --threads 4 --timeout 120 -b 127.0.0.1:5002 julia_server:app
```

systemd: dua unit (`laya-arena.service`, `julia-arena.service`) dengan pola
yang sama seperti sebelumnya — `Environment=USE_TF=0` untuk laya,
`WorkingDirectory` repo, `Restart=always`, 1 worker per unit. Julia cukup
bind `127.0.0.1` ( hanya diakses proxy dari laya server ).
