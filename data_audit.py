#!/usr/bin/env python3
"""
data_audit.py  (Giai doan D1 - khao sat chat luong audio)
=========================================================
Quet dataset InFoRe, do cac chi so chat luong (KHONG sua gi):
  - Loudness (RMS dBFS) - phan bo, do dong deu am luong
  - Peak / Clipping      - file bi vo tieng
  - DC offset            - lech 0
  - Khoang lang dau/cuoi - bao nhieu giay im lang 2 dau
  - Khoang lang GIUA cau - so diem nghi noi bo (de chuan bi chen dau cau o D3)
  - Do dai               - phan bo

Cach dung:
    python data_audit.py                  # mau (1/10 file, nhanh)
    python data_audit.py --all            # toan bo 14835 file (cham hon)
    python data_audit.py --every 5        # mau 1/5
"""

import argparse
import os
import statistics
import wave
from pathlib import Path

import numpy as np

DATA_DIR = r"dataset/infore_16k_denoised_2"
SIL_THRESH = 0.01        # nguong bien do coi la "lang"
MIN_PAUSE_SEC = 0.20     # khoang lang >= 0.2s giua cau -> tinh la 1 diem nghi


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def analyze(x, sr):
    if len(x) == 0:
        return None
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    peak = np.max(np.abs(x)) + 1e-9
    dc = float(np.mean(x))
    rms_db = 20 * np.log10(rms)
    peak_db = 20 * np.log10(peak)
    clip = int(np.sum(np.abs(x) >= 0.99))

    # lang dau/cuoi
    above = np.abs(x) > SIL_THRESH
    if above.any():
        first, last = np.argmax(above), len(x) - np.argmax(above[::-1])
        lead_sil = first / sr
        trail_sil = (len(x) - last) / sr
    else:
        lead_sil = trail_sil = len(x) / sr

    # khoang lang GIUA (diem nghi noi bo) - dem cac doan im >= MIN_PAUSE_SEC
    # (bo qua lang dau/cuoi)
    win = int(0.02 * sr)  # khung 20ms
    n_win = len(x) // win
    energy = np.array([np.sqrt(np.mean(x[i*win:(i+1)*win]**2)) for i in range(n_win)])
    silent = energy < SIL_THRESH
    pauses, run = 0, 0
    started = False
    for s in silent:
        if s:
            run += 1
        else:
            if started and run * 0.02 >= MIN_PAUSE_SEC:
                pauses += 1
            run = 0
            started = True
    # khong tinh lang cuoi
    return dict(rms_db=rms_db, peak_db=peak_db, dc=dc, clip=clip,
               lead_sil=lead_sil, trail_sil=trail_sil, pauses=pauses,
               dur=len(x)/sr)


def hist(vals, bins, labels):
    counts = [0] * len(labels)
    for v in vals:
        for i, b in enumerate(bins):
            if v < b:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    total = len(vals)
    for lab, c in zip(labels, counts):
        bar = "#" * (c * 40 // total) if total else ""
        print(f"    {lab:>12}: {c:5d} ({c*100//max(total,1):2d}%) {bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DATA_DIR)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()

    ids = sorted(f[:-4] for f in os.listdir(args.dir) if f.endswith(".wav"))
    if not args.all:
        ids = ids[::args.every]
    print(f"[info] Phan tich {len(ids)} file (toan bo: {args.all})...\n")

    R = {k: [] for k in ["rms_db", "peak_db", "dc", "lead_sil", "trail_sil", "pauses", "dur"]}
    n_clip = 0
    for fid in ids:
        try:
            x, sr = read_wav(os.path.join(args.dir, fid + ".wav"))
            a = analyze(x, sr)
            if a is None:
                continue
            for k in R:
                R[k].append(a[k])
            if a["clip"] > 10:
                n_clip += 1
        except Exception as e:
            print(f"  [loi] {fid}: {e}")

    def stat(k):
        v = R[k]
        return f"min={min(v):.2f} max={max(v):.2f} tb={statistics.mean(v):.2f} median={statistics.median(v):.2f}"

    print("=" * 60)
    print("1) LOUDNESS (RMS dBFS) - cang dong deu cang tot")
    print(f"   {stat('rms_db')}")
    print(f"   Do lech chuan: {statistics.pstdev(R['rms_db']):.2f} dB  (lon = am luong khong deu)")
    hist(R["rms_db"], [-40, -35, -30, -25, -20, -15],
         ["<-40", "-40..-35", "-35..-30", "-30..-25", "-25..-20", "-20..-15", ">-15"])

    print("\n2) PEAK (dBFS) + CLIPPING")
    print(f"   peak: {stat('peak_db')}")
    print(f"   File nghi clipping (>10 mau >=0.99): {n_clip} ({n_clip*100//len(ids)}%)")

    print("\n3) DC OFFSET (nen gan 0)")
    print(f"   {stat('dc')}")

    print("\n4) KHOANG LANG DAU (giay)")
    print(f"   {stat('lead_sil')}")
    hist(R["lead_sil"], [0.05, 0.1, 0.2, 0.3, 0.5],
         ["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", ">0.5"])

    print("\n5) KHOANG LANG CUOI (giay)")
    print(f"   {stat('trail_sil')}")
    hist(R["trail_sil"], [0.05, 0.1, 0.2, 0.3, 0.5],
         ["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", ">0.5"])

    print(f"\n6) DIEM NGHI NOI BO (lang >= {MIN_PAUSE_SEC}s giua cau) - de chen dau cau o D3")
    print(f"   {stat('pauses')}")
    hist(R["pauses"], [1, 2, 3, 4],
         ["0", "1", "2", "3", ">=4"])

    print("\n7) DO DAI (giay)")
    print(f"   {stat('dur')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
