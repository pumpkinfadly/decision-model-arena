# laya-multilingual playground

Web playground for [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) —
322M non-autoregressive decision/routing model. Given state (text / email /
ticket / transcript) + typed questions, returns typed answers with
probabilities in one forward pass. No text generation. Apache 2.0.

## Menjalankan lokal (Windows)

```cmd
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\python.exe app.py
```

→ http://127.0.0.1:5001

Weights di-cache ke `%USERPROFILE%\.cache\huggingface\hub\` (~775 MB) saat
download pertama.

Tes CLI: `.venv\Scripts\python.exe test_laya.py` ·
`.venv\Scripts\python.exe run_sample.py [sample_key]`

## Fitur

- Contoh berpasangan: 1 chip = state + pertanyaan sekaligus (ID/EN, domain ISP)
- Upload JSON: state (`{"transcript": [...]}` → digabung jadi body, atau
  `{"body": "..."}`) + questions (`{qid: {type, instructions, criteria}}`,
  nilai null → "")
- Question builder: `choice` (opsi), `score` (skala berurut), `noul` (ya/tidak)
  — plus preset laya (triage/email/router/moderation)
- Hasil: bar probabilitas, **top-3 di-highlight kalau opsi > 5**, confidence,
  peringatan low-confidence, token usage
- API: `POST /api/predict`, `GET /api/presets`, `GET /api/samples`

## Data contoh

- `data/simulasi_chat_gangguan_internet.json` — state: simulasi chat ISP
  Indonesia (transcript + taksonomi kategori 4 level)
- `data/contoh_pertanyaan_category.json` (v1) — 1 pertanyaan `category`,
  34 opsi full path `Level1 > Level2 > Level3 > Level4`
- `data/contoh_pertanyaan_category_v2.json` (v2) — sama, tapi opsi = nama
  kategori inti saja tanpa path
- Chip "Contoh: ISP ..." (v1) dan "Contoh v2: ISP ..." memuat keduanya

Hasil zero-shot pada transcript yang sama:

| versi | top-1 | prob | confidence |
|---|---|---|---|
| v1 full path | Cek Status Gangguan | 8.3% | 4.1% |
| v2 kategori inti | Internet Tidak Bisa Digunakan | 39.8% | 49.4% |

Catatan: 34 opsi tetap melebihi batas ~20 opsi per `choice` di model card —
pecah per level untuk hasil terbaik (set bertingkat pernah benar semua dengan
confidence 51–98%).

## Deploy di server (Ubuntu/Debian)

Kebutuhan: 2 vCPU, RAM 2GB **minimal + swap wajib** (proses app ~1.7 GB,
tidak ada opsi half/quant di laya 0.3.7), disk ≥ 3 GB, Python 3.10+.
Saat trafik naik: upgrade 4 GB.

```bash
# 1. swap 2 GB (wajib untuk RAM 2 GB)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 2. dependensi sistem
sudo apt update && sudo apt install -y python3-venv git

# 3. clone + virtualenv
git clone https://github.com/pumpkinfadly/laya-multilingual-playground.git
cd laya-multilingual-playground
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install gunicorn

# 4. pre-download model (~775 MB) agar start pertama tidak timeout
USE_TF=0 python -c "import laya; laya.load('convaiinnovations/laya-multilingual')"

# 5. tes jalan
USE_TF=0 gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:5001 app:app
```

### systemd service

`/etc/systemd/system/laya-playground.service` (sesuaikan user + path):

```ini
[Unit]
Description=laya-multilingual playground
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/laya-multilingual-playground
Environment=USE_TF=0
ExecStart=/home/ubuntu/laya-multilingual-playground/.venv/bin/gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:5001 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now laya-playground
sudo systemctl status laya-playground
```

Penting: **1 worker saja** (`-w 1`) — model dimuat per proses (~1.7 GB) dan
backend memakai lock; threads aman untuk request paralel ringan. Akses port
5001 via reverse proxy (nginx/caddy) kalau perlu HTTPS.
