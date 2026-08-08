#!/usr/bin/env python3
"""
demo_vits.py
============
Sinh audio demo tu checkpoint VITS da train.

KIEN TRUC (da tach gon):
  - Chuan hoa van ban  -> module vi_normalize.py  (so/viet tat/ngay/toan...)
  - Cau demo           -> file data demo/sentences.txt  (sua o do, khong dong code)
  - File nay           -> chi lo: tai model, tach cau ngat nghi, cat lang, fade, ghep.

Cach dung:
    python demo_vits.py
    python demo_vits.py --ckpt "models/vits/runs/vits_infore-June-19-2026_06+31PM-0000000/checkpoint_100068.pth" --out-dir demo/vits_switch
    python demo_vits.py --ckpt ".../best_model.pth" --out-dir demo/vits_best
    python demo_vits.py --sentences demo/sentences.txt
    python demo_vits.py --no-pause            # doc lien 1 lan
    python demo_vits.py --sil-short 0.15 --sil-long 0.25 --trim-thresh 0.02
"""

import argparse
import time
from pathlib import Path

import numpy as np

from vi_normalize import normalize, split_for_pause, strip_punct

RUN_DIR = "models/vits/runs/vits_infore-June-19-2026_06+31PM-0000000"


def load_sentences(path):
    """Doc cau demo tu file text (bo dong trong / dong #)."""
    lines = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{RUN_DIR}/best_model.pth")
    ap.add_argument("--config", default=f"{RUN_DIR}/config.json")
    ap.add_argument("--out-dir", default="demo/vits_100k")
    ap.add_argument("--sentences", default="demo/sentences.txt",
                    help="File text chua cau demo (moi dong 1 cau)")
    ap.add_argument("--no-pause", action="store_true",
                    help="Tong hop ca cau 1 lan (khong chen khoang lang)")
    ap.add_argument("--sil-short", type=float, default=0.2, help="Nghi dau phay , ; : (s)")
    ap.add_argument("--sil-long", type=float, default=0.3, help="Nghi dau cham . ! ? (s)")
    ap.add_argument("--sil-none", type=float, default=0.1, help="Nghi mac dinh giua doan (s)")
    ap.add_argument("--trim-thresh", type=float, default=0.015,
                    help="Nguong cat lang dau/cuoi moi doan (0=tat)")
    args = ap.parse_args()

    if not Path(args.ckpt).exists():
        raise SystemExit(f"[error] Khong tim thay checkpoint: {args.ckpt}")
    if not Path(args.sentences).exists():
        raise SystemExit(f"[error] Khong tim thay file cau: {args.sentences}")

    sentences = load_sentences(args.sentences)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from TTS.api import TTS
    from TTS.config import load_config
    import soundfile as sf

    print(f"[load] Checkpoint: {args.ckpt}")
    tts = TTS(model_path=args.ckpt, config_path=args.config)
    tts.to("cuda")
    sr = load_config(args.config).audio.sample_rate
    print(f"[info] {len(sentences)} cau | {sr} Hz | Ngat nghi: {'TAT' if args.no_pause else 'BAT'}\n")

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
        import io
        buf = io.BytesIO()
        tts.tts_to_file(text=text, file_path=buf)
        buf.seek(0)
        wav, _ = sf.read(buf, dtype="float32")
        return apply_fade(trim_sil(wav))

    SIL = {"long": args.sil_long, "short": args.sil_short, "none": args.sil_none}

    total_audio, total_wall = 0.0, 0.0
    for i, raw in enumerate(sentences, 1):
        norm = normalize(raw)
        out_wav = out_dir / f"demo{i}.wav"
        t0 = time.time()

        if args.no_pause:
            audio = synth(strip_punct(norm))
        else:
            segs = split_for_pause(norm)
            chunks = []
            for j, (seg, kind) in enumerate(segs):
                chunks.append(synth(seg))
                if j < len(segs) - 1:
                    chunks.append(np.zeros(int(SIL[kind] * sr), dtype=np.float32))
            audio = np.concatenate(chunks)

        sf.write(str(out_wav), audio, sr)
        elapsed = time.time() - t0
        dur = len(audio) / sr
        total_audio += dur
        total_wall += elapsed
        n_seg = 1 if args.no_pause else len(split_for_pause(norm))
        print(f"[{i}] {dur:4.1f}s / {n_seg} doan -> {out_wav}")
        print(f"    raw : {raw[:65]}")
        print(f"    norm: {norm[:65]}")

    rtf = total_wall / total_audio if total_audio else 0
    print(f"\n[done] {len(sentences)} file trong: {out_dir}")
    print(f"       Tong audio: {total_audio:.1f}s | xu ly: {total_wall:.1f}s | RTF = {rtf:.3f}")


if __name__ == "__main__":
    main()
