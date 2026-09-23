# Alur Backend — Playground laya-multilingual

Dokumen ini menjelaskan cara kerja backend aplikasi, khususnya mode klasifikasi
bertingkat (`/api/predict_hierarchical`). Semua kode ada di `app.py`.

## Gambaran umum

Backend adalah aplikasi Flask tunggal yang memuat model
[convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)
(322M parameter, non-autoregressive) sekali saat startup, lalu menyediakan
beberapa endpoint JSON yang dipanggil frontend. Tidak ada database; seluruh
data contoh dibaca dari file JSON di folder `data/`.

## Saat server dinyalakan (sekali saja)

1. **Muat model** — `laya.load("convaiinnovations/laya-multilingual")` memuat
   seluruh bobot model ke memori (sekitar 1,7 GB pada CPU). Model dibuat satu
   instance global dan dipakai bersama semua request. Karena model torch tidak
   thread-safe untuk pemakaian bersamaan, setiap panggilan `predict` dibungkus
   `predict_lock` (threading lock) sehingga hanya satu inferensi berjalan pada
   satu waktu; request lain menunggu.

2. **Bangun taksonomi** — `_build_taxonomy()` membaca
   `data/simulasi_chat_gangguan_internet.json`, mengambil larik
   `kategori_level`, lalu menurunkan peta `TAXONOMY_L34` berformat
   `{level_3: [daftar level_4, ...]}`. Contoh isinya:

   ```python
   {
     "Gangguan Internet": ["Cek Status Gangguan", "Request Teknisi",
                            "Internet Tidak Bisa Digunakan", "Internet Putus-Putus",
                            "Koneksi Lambat", "LOS Merah", "Tidak Ada Sinyal"],
     "Paket Internet": ["Informasi Harga Paket", "Informasi Benefit Paket",
                         "Upgrade Paket", "Downgrade Paket"],
     ...
   }
   ```

   File JSON ini menjadi satu-satunya sumber taksonomi. Jika kategori
   ditambah/diubah di file tersebut, cukup restart server — pertanyaan level 4
   otomatis mengikuti tanpa perubahan kode.

3. **Siapkan definisi pertanyaan statis** — `HIER_L1`, `HIER_L2`, `HIER_L3`:
   masing-masing pertanyaan `choice` dengan jumlah opsi kecil (3 / 5 / 11) dan
   deskripsi deskriminatif untuk setiap opsi. Ini mengikuti panduan model card
   (jaga opsi di bawah ±20; idealnya 3–11) agar softmax antar opsi tetap tajam.

4. **Muat sampel** — `load_samples()` menyiapkan tiga contoh berpasangan
   (state + pertanyaan) dari file di `data/`:
   - v1: opsi kategori berformat path penuh `Level1, Level2, Level3, Level4`
   - v2: opsi hanya nama kategori inti (level 4)
   - v3: mode bertingkat (pertanyaan dibuat otomatis server, bukan dari file)

## Endpoint

| Endpoint | Fungsi |
|---|---|
| `GET /` | Halaman playground |
| `POST /api/predict` | Predict biasa: state + pertanyaan dari builder frontend |
| `POST /api/predict_hierarchical` | Predict bertingkat dua tahap (lihat bawah) |
| `GET /api/presets` | Set pertanyaan bawaan paket laya (triage, email, router, moderation) |
| `GET /api/samples` | Daftar contoh berpasangan untuk chip di frontend |

## Alur `/api/predict_hierarchical`

Request hanya berisi `{"body": "<teks percakapan>"}`. Endpoint menjalankan
dua tahap inferensi secara berurutan:

```
body (transcript pelanggan)
  │
  ├─ Validasi
  │    body kosong            → 400 "'body' harus diisi"
  │    TAXONOMY_L34 kosong    → 500 (file data/ hilang)
  │
  ▼
TAHAP 1 — satu panggilan agent.predict()
  state     = {"body": body}
  pertanyaan = {
    level_1: HIER_L1  (3 opsi : Informasi / Permintaan / Komplain),
    level_2: HIER_L2  (5 opsi : Internet / Billing / Akun / Instalasi / Teknisi),
    level_3: HIER_L3  (11 opsi : sub-kategori),
  }
  → tiga pertanyaan diproses dalam SATU forward pass
  → jawaban contoh: Komplain / Internet / Gangguan Internet
  │
  ├─ Ambil cabang: leaves = TAXONOMY_L34["Gangguan Internet"]  → 7 opsi
  │  (jika cabang tidak punya daun, tahap 2 dilewati dan path berhenti di level 3)
  │
  ▼
TAHAP 2 — panggilan agent.predict() kedua
  state      = {"body": body + "\n\nKonteks klasifikasi sebelumnya:
                         Komplain > Internet > Gangguan Internet"}
               ↑ hasil tahap 1 disuntikkan ke body sebagai konteks teks
  pertanyaan = {level_4: choice dengan criteria = 7 daun cabang terpilih}
  → jawaban contoh: Internet Tidak Bisa Digunakan (confidence 49%)
  │
  ▼
Respons JSON
{
  "answers": {level_1..level_4: {choice, probabilities, confidence, act_probability}},
  "final_path": "Komplain > Internet > Gangguan Internet > Internet Tidak Bisa Digunakan",
  "usage": {"input_tokens": 1992},        // tahap 1 + tahap 2 dijumlahkan
  "detected_language": "id"
}
```

### Contoh hasil nyata (transcript gangguan internet)

| Level | Jawaban | Confidence |
|---|---|---|
| level_1 | Komplain | 97,0% |
| level_2 | Internet | 98,5% |
| level_3 | Gangguan Internet | 68,8% |
| level_4 | Internet Tidak Bisa Digunakan | 49,0% |

## Keputusan desain

1. **Conditioning lewat teks, bukan parameter.** Tahap 2 tidak mengubah model
   atau parameter apa pun — hasil tahap 1 cukup ditambahkan sebagai kalimat
   konteks di body. Model yang sama tetap dipakai untuk semua cabang.

2. **Opsi tahap 2 dinamis.** Daftar opsi level 4 diambil dari pohon taksonomi
   sesuai jawaban level 3, sehingga jumlah opsi selalu kecil (1–7 pada
   taksonomi saat ini). Inilah penyebab utama mode bertingkat jauh lebih akurat
   daripada satu pertanyaan 34 opsi (confidence 4% → 49–99%).

3. **Dua forward pass, bukan satu.** Model non-autoregressive sehingga setiap
   pass hanya berupa klasifikasi cepat (total sekitar setengah detik pada CPU);
   biaya tahap kedua sangat murah dibanding keuntungan akurasi.

4. **Satu lock global.** `predict_lock` membungkus tiap `agent.predict()` di
   kedua tahap sehingga request paralel tidak merusak state model. Frontend
   mengirim request satu-satu; lock berfungsi sebagai pengaman tambahan saat
   dipakai lewat API langsung.

5. **Format respons seragam.** `_normalize_answers()` dipakai bersama oleh
   `/api/predict` dan `/api/predict_hierarchical`, sehingga frontend cukup punya
   satu fungsi render untuk semua mode. Mode bertingkat hanya menambah field
   `final_path`.

6. **Error per tahap terpisah.** Kegagalan tahap 1 dan tahap 2 mengembalikan
   pesan berbeda (`predict tahap 1 gagal: ...` / `predict tahap 2 gagal: ...`)
   sehingga penyebab mudah dilokalisasi; tidak ada respons setengah jadi.

## Perbandingan tiga desain contoh (transcript yang sama)

| Desain | Opsi per pertanyaan | Hasil | Confidence |
|---|---|---|---|
| v1 — path penuh | 34 | salah (berubah-ubah antar format) | ±4% |
| v2 — kategori inti | 34 | benar | 49% |
| v3 — bertingkat | 3 / 5 / 11 / 7 | benar di semua level | 49–99% |

Kesimpulan: jumlah opsi per pertanyaan dan kualitas label menentukan akurasi;
format path panjang mengencerkan sinyal pembeda karena tiga dari empat segmen
labelnya sama untuk banyak opsi.
