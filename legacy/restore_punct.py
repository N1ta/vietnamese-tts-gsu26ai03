#!/usr/bin/env python3
"""
restore_punct.py  (Giai doan D3 - khoi phuc dau cau tu khoang lang audio)
=========================================================================
Chen dau cau vao transcript DUA TREN khoang lang THAT trong audio:
  - Phat hien khoang lang >= MIN_PAUSE giua cau -> audio chia thanh CAC CUM.
  - So cum = so lan nghi -> so dau phay can chen.
  - Chia chu trong transcript theo TI LE THOI LUONG moi cum -> vi tri phay xap xi.
  - GIU NGUYEN chu goc, chi them dau phay (giua) + dau cham (cuoi).

Cach dung:
    python restore_punct.py --demo 12          # xem ket qua tren 12 file (KHONG ghi)
    python restore_punct.py --demo 12 --ids 00000 00100   # xem file cu the
    python restore_punct.py --out dataset/infore_coqui/metadata_punct.csv  # ghi toan bo
"""

import argparse
import os
import wave
from pathlib import Path

import numpy as np

DATA_DIR = r"dataset/infore_16k_denoised_2"
MIN_PAUSE = 0.25     # khoang lang >= 0.25s -> 1 diem nghi (chen phay)
SIL_THRESH = 0.012   # nguong bien do coi la "lang"
WIN = 0.02           # khung 20ms


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def voiced_segments(x, sr):
    """Tra ve list (start_s, end_s) cac cum phat am, va list khoang lang giua chung."""
    win = int(WIN * sr)
    n_win = len(x) // win
    energy = np.array([np.sqrt(np.mean(x[i*win:(i+1)*win]**2)) for i in range(n_win)])
    voiced = energy >= SIL_THRESH

    # tim cac doan voiced lien tuc
    segs = []
    i = 0
    while i < n_win:
        if voiced[i]:
            j = i
            while j < n_win and voiced[j]:
                j += 1
            segs.append([i * WIN, j * WIN])
            i = j
        else:
            i += 1
    if not segs:
        return [], []

    # gop 2 cum neu khoang lang giua < MIN_PAUSE (lang ngan khong tinh la nghi)
    merged = [segs[0]]
    gaps = []
    for s in segs[1:]:
        gap = s[0] - merged[-1][1]
        if gap < MIN_PAUSE:
            merged[-1][1] = s[1]   # gop
        else:
            gaps.append(gap)
            merged.append(s)
    return merged, gaps


def insert_punct(text, n_segs, seg_durs):
    """Chia 'text' thanh n_segs cum theo ti le thoi luong, noi bang ', ', cuoi them '.'."""
    words = text.split()
    if n_segs <= 1 or len(words) <= n_segs:
        return text.strip() + "."
    total = sum(seg_durs)
    # so tu cho moi cum theo ti le thoi luong
    counts = [max(1, round(len(words) * d / total)) for d in seg_durs]
    # chinh lai cho tong = len(words)
    while sum(counts) > len(words):
        counts[counts.index(max(counts))] -= 1
    while sum(counts) < len(words):
        counts[counts.index(min(counts))] += 1
    parts, idx = [], 0
    for c in counts:
        parts.append(" ".join(words[idx:idx + c]))
        idx += c
    return ", ".join(p for p in parts if p) + "."


def process(fid, data_dir):
    x, sr = read_wav(os.path.join(data_dir, fid + ".wav"))
    text = Path(os.path.join(data_dir, fid + ".txt")).read_text(encoding="utf-8").strip()
    segs, gaps = voiced_segments(x, sr)
    durs = [e - s for s, e in segs]
    punct = insert_punct(text, len(segs), durs)
    return text, punct, len(segs), gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DATA_DIR)
    ap.add_argument("--demo", type=int, default=0, help="So file demo (khong ghi)")
    ap.add_argument("--ids", nargs="*", help="ID cu the de demo")
    ap.add_argument("--out", default=None, help="Ghi metadata 3 cot ra file (toan bo)")
    args = ap.parse_args()

    all_ids = sorted(f[:-4] for f in os.listdir(args.dir) if f.endswith(".wav"))

    if args.demo or args.ids:
        ids = args.ids if args.ids else all_ids[::max(1, len(all_ids)//args.demo)][:args.demo]
        for fid in ids:
            text, punct, nseg, gaps = process(fid, args.dir)
            gstr = ", ".join(f"{g:.2f}s" for g in gaps) or "(khong co)"
            print(f"[{fid}] {nseg} cum, nghi: {gstr}")
            print(f"   goc  : {text}")
            print(f"   punct: {punct}\n")
        return

    if args.out:
        lines = []
        for k, fid in enumerate(all_ids):
            try:
                text, punct, _, _ = process(fid, args.dir)
                lines.append(f"{fid}|{text}|{punct}")
            except Exception as e:
                print(f"  [loi] {fid}: {e}")
            if (k + 1) % 1000 == 0:
                print(f"  ...{k+1}/{len(all_ids)}")
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[done] Ghi {len(lines)} dong -> {args.out}")
        return

    ap.error("Chon --demo N hoac --ids ... hoac --out file")


if __name__ == "__main__":
    main()
