#!/usr/bin/env python3
"""
infer_long_text.py
==================
Doc van ban TIENG VIET DAI (>2000 tu) bang model VITS da train.
(Yeu cau giam thi #1 — chunking van ban dai.)

DUNG CHUNG module chuan hoa vi_normalize (giong demo_vits): so/viet tat/ngay/
phan so/toan + tach cau ngat nghi + cat lang + fade chong click.

LOGIC CHUNKING (giai thich cho bao cao):
  - Model TTS co gioi han do dai ngo vao (attention O(n^2), positional enc).
  - Cat van ban thanh CHUNK theo dau cau (. ! ? ; ,), moi chunk <= max_chars.
  - Tong hop tung chunk roi GHEP audio (numpy concat) + khoang lang giua chunk.
  - Cat o dau cau => ngat nghi tu nhien, khong vo nghia giua cau.

Cach dung:
    python infer_long_text.py --text bai_dai.txt --out output_long.wav
    python infer_long_text.py --text demo/bai_dai.txt --ckpt "models/vits/runs/vits_infore-June-19-2026_06+31PM-0000000/checkpoint_100068.pth" --out demo/test_dai.wav
    python infer_long_text.py --text "doan van ban..." --out test.wav
    python infer_long_text.py --dry-run --text bai_dai.txt   # chi xem chunk
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from vi_normalize import normalize, split_for_pause

RUN_DIR = "models/vits/runs/vits_infore-June-19-2026_06+31PM-0000000"
MAX_CHARS = 200  # gioi han ky tu moi chunk (VITS)


def make_units(segs, max_chars):
    """Tao danh sach don vi tong hop [(text, kind)] tu cac doan da tach o dau cau.
    QUAN TRONG: GIU nghi sau MOI doan (dau cau) -> khong mat nghi o dau cham.
    Chi cat nho khi 1 doan > max_chars (cat theo tu, nghi 'none' giua cac manh)."""
    units = []
    for text, kind in segs:
        if len(text) <= max_chars:
            units.append((text, kind))
            continue
        # doan qua dai -> cat theo tu thanh nhieu manh
        words = text.split()
        piece = ""
        pieces = []
        for w in words:
            if len(piece) + len(w) + 1 <= max_chars:
                piece = (piece + " " + w).strip()
            else:
                pieces.append(piece)
                piece = w
        if piece:
            pieces.append(piece)
        for i, p in enumerate(pieces):
            # manh cuoi giu 'kind' goc, cac manh truoc nghi ngan 'none'
            units.append((p, kind if i == len(pieces) - 1 else "none"))
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Van ban hoac duong dan .txt")
    ap.add_argument("--out", default="output_long.wav")
    ap.add_argument("--ckpt", default=f"{RUN_DIR}/best_model.pth")
    ap.add_argument("--config", default=f"{RUN_DIR}/config.json")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--sil-short", type=float, default=0.2)
    ap.add_argument("--sil-long", type=float, default=0.3)
    ap.add_argument("--sil-none", type=float, default=0.1)
    ap.add_argument("--trim-thresh", type=float, default=0.015)
    ap.add_argument("--dry-run", action="store_true", help="Chi in chunk, khong tong hop")
    args = ap.parse_args()

    # Doc text (file hoac chuoi)
    text_in = args.text
    if Path(text_in).exists():
        text_in = Path(text_in).read_text(encoding="utf-8")

    # Chuan hoa + tach doan o dau cau (GIU nghi tung dau cau)
    norm = normalize(text_in)
    segs = split_for_pause(norm)
    chunks = make_units(segs, args.max_chars)

    n_words = len(norm.split())
    print(f"[info] Van ban: {len(norm)} ky tu, ~{n_words} tu")
    print(f"[info] {len(segs)} doan (tach o dau cau) -> {len(chunks)} don vi tong hop (<= {args.max_chars} ky tu)")
    for i, (c, k) in enumerate(chunks, 1):
        print(f"  [{i:3d}] ({len(c):3d}) [{k:5s}] {c[:64]}{'...' if len(c) > 64 else ''}")

    if args.dry_run:
        print("\n[dry-run] Xong, khong tong hop.")
        return

    if not Path(args.ckpt).exists():
        sys.exit(f"[error] Khong tim thay checkpoint: {args.ckpt}")

    from TTS.api import TTS
    from TTS.config import load_config
    import soundfile as sf
    import io

    print(f"\n[load] {args.ckpt}")
    tts = TTS(model_path=args.ckpt, config_path=args.config)
    tts.to("cuda")
    sr = load_config(args.config).audio.sample_rate

    def trim_sil(wav, pad=0.02):
        if args.trim_thresh <= 0 or len(wav) == 0:
            return wav
        idx = np.where(np.abs(wav) > args.trim_thresh)[0]
        if len(idx) == 0:
            return wav
        p = int(pad * sr)
        return wav[max(0, idx[0] - p):min(len(wav), idx[-1] + p)]

    def apply_fade(wav, fade_ms=10):
        n = min(int(fade_ms / 1000 * sr), len(wav) // 2)
        if n > 0:
            wav = wav.copy()
            wav[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
            wav[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
        return wav

    def synth(text):
        buf = io.BytesIO()
        tts.tts_to_file(text=text, file_path=buf)
        buf.seek(0)
        wav, _ = sf.read(buf, dtype="float32")
        return apply_fade(trim_sil(wav))

    SIL = {"long": args.sil_long, "short": args.sil_short, "none": args.sil_none}

    pieces = []
    t0 = time.time()
    for i, (chunk, kind) in enumerate(chunks):
        pieces.append(synth(chunk))
        if i < len(chunks) - 1:
            pieces.append(np.zeros(int(SIL[kind] * sr), dtype=np.float32))
        print(f"  chunk {i+1}/{len(chunks)} xong")

    audio = np.concatenate(pieces)
    sf.write(args.out, audio, sr)
    dur = len(audio) / sr
    elapsed = time.time() - t0
    print(f"\n[done] {args.out}")
    print(f"       Audio: {dur:.1f}s | xu ly: {elapsed:.1f}s | RTF: {elapsed/dur:.3f}")
    print(f"       {len(chunks)} chunk, TB {sum(len(c) for c,_ in chunks)//len(chunks)} ky tu/chunk")


if __name__ == "__main__":
    main()
