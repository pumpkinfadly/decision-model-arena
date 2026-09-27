"""One-off setup: download Julia-1 snapshot to julia_model/ (local dir)."""
from huggingface_hub import snapshot_download

path = snapshot_download("SupersonicLabs/Julia-1", local_dir="julia_model")
print("downloaded to:", path)
