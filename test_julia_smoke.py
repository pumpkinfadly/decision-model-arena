"""Smoke test: load Julia-1 engine, one choice + one noul predict."""
import time

from julia import load_model

t0 = time.perf_counter()
engine = load_model("julia_model")
print(f"loaded in {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
answers = engine.predict(
    state="Halo, internet saya dari tadi pagi tidak bisa dipakai. Lampu LOS di modem merah.",
    questions={
        "tes": {
            "instructions": "Tim mana yang harus menangani laporan ini?",
            "type": "choice",
            "criteria": {
                "gangguan": "internet mati, lambat, lampu LOS merah",
                "billing": "tagihan, pembayaran, refund",
                "penjualan": "paket baru, harga, promo",
            },
        },
        "refund": {
            "instructions": "Apakah pelanggan meminta uang dikembalikan?",
            "type": "noul",
            "criteria": None,
        },
    },
)
print(f"predict in {(time.perf_counter() - t0) * 1000:.0f} ms")
print(answers)
