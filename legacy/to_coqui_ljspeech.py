#!/usr/bin/env python3
"""
to_coqui_ljspeech.py
====================
Chuyen metadata 2 cot (wavs/<id>.wav|text) sang dinh dang LJSpeech 3 cot ma
Coqui formatter `ljspeech` yeu cau:

    <id>|<raw_text>|<normalized_text>

Trong do:
  - <id>  : ten file KHONG co duong dan, KHONG co duoi .wav
            (Coqui tu them .wav va tu tim trong <path>/wavs/)
  - cot 2 va cot 3: deu la text (Coqui doc cot thu 3 lam transcript)

Cach dung:
    python to_coqui_ljspeech.py ^
        --in  dataset/single_speaker/VIVOSSPK35/metadata_char.csv ^
        --out dataset/single_speaker/VIVOSSPK35/metadata_coqui.csv

Sau do trong coqui_config.json doi:
    "meta_file_train": "metadata_coqui.csv"
va dam bao wav nam o:  <path>/wavs/<id>.wav  (script prepare da tao thu muc wavs/)
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    args = ap.parse_args()

    n_in, n_out, skipped = 0, 0, 0
    lines = []
    for line in Path(args.infile).read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        n_in += 1
        parts = line.split("|")
        if len(parts) < 2:
            skipped += 1
            continue
        wav_field = parts[0].strip()
        text = "|".join(parts[1:]).strip()
        # PHAT HIEN text da bi token hoa (co '<sp>' hoac ky tu cach roi).
        # Coqui can text THUONG, khong phai 'n h u <sp>'. Neu thay <sp>, khoi phuc.
        if "<sp>" in text:
            # 'n h ư n g <sp> c ả' -> 'nhưng cả'
            words = text.split("<sp>")
            text = " ".join("".join(w.split()) for w in words).strip()
        # bo duong dan + duoi .wav -> chi giu id
        stem = Path(wav_field).name
        if stem.lower().endswith(".wav"):
            stem = stem[:-4]
        if not text:
            skipped += 1
            continue
        # 3 cot: id|raw|normalized (dung cung text cho ca 2 cot)
        lines.append(f"{stem}|{text}|{text}")
        n_out += 1

    Path(args.outfile).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] doc {n_in} dong, ghi {n_out} dong (3 cot) -> {args.outfile}")
    if skipped:
        print(f"[warn] bo qua {skipped} dong thieu text/cot")
    print("Nho doi 'meta_file_train' trong coqui_config.json thanh ten file nay.")
    print("Va kiem tra wav nam o: <path>/wavs/<id>.wav")


if __name__ == "__main__":
    main()