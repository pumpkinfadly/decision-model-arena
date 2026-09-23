r"""Convert raw DB category paths to a hierarchical question config.

Input file: JSON list of comma path strings, {"paths": [...]}, or a list of
{level_1: ..., level_2: ...} objects.

Usage:
    python tools/convert_from_db.py input.json [output.json]

Without output path, writes <input>_hierarki.json next to the input.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hier_converter import convert  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_hierarki.json"
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    cfg = convert(data)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    levels = cfg["levels"]
    print(f"OK: {len(cfg.get('paths', [])) or ''} -> {dst}")
    for qid, q in levels.items():
        if "branches" in q:
            print(f"  {qid}: {len(q['branches'])} cabang, "
                  f"{sum(len(b) for b in q['branches'].values())} leaf")
        else:
            print(f"  {qid}: {len(q['criteria'])} opsi")
    print("Catatan: deskripsi level atas di-generate dari anak tiap node — "
          "edit manual untuk akurasi terbaik.")


if __name__ == "__main__":
    main()
