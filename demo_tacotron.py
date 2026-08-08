#!/usr/bin/env python3
"""
demo_tacotron.py
================
Sinh audio demo tu Tacotron2 + HiFi-GAN vocoder (InFoRe 16kHz).

KHAC demo_vits.py:
  - VITS la end-to-end (text -> waveform). Tacotron2 chi sinh MEL -> can vocoder.
  - Dung Coqui `Synthesizer`: nap CA Tacotron2 (acoustic) + HiFi-GAN (vocoder),
    text -> mel (Tacotron2) -> waveform (HiFi-GAN).
  - Tacotron2 train tren transcript GOC (khong dau cau, vocab chi co chu + space)
    => PHAI strip dau cau tung doan truoc khi tong hop; van giu ngat nghi bang
    cach tach doan o dau cau + chen khoang lang (giong demo_vits).

Chuan hoa van ban dung chung module vi_normalize (so/viet tat/ngay/toan...).

Cach dung:
    python demo_tacotron.py
    python demo_tacotron.py --out-dir demo/tacotron_hifigan
    python demo_tacotron.py --sentences demo/sentences.txt
    python demo_tacotron.py --voc-ckpt ".../checkpoint_68572.pth"
    python demo_tacotron.py --no-pause
"""

import argparse
import time
from pathlib import Path

import numpy as np

from vi_normalize import normalize, split_for_pause, strip_punct

# Checkpoint mac dinh (sua o day neu doi run)
TAC_RUN = "models/tacotron2/runs/tacotron2_infore_punct-June-26-2026_08+39PM-0000000"
HIFI_RUN = "models/hifigan/runs/hifigan_infore-June-27-2026_07+33PM-0000000"


def make_units(segs, max_chars):
    """Gom cac doan (da tach o dau cau) thanh don vi <= max_chars de tong hop.
    Doan qua dai -> cat theo tu; cac manh truoc nghi ngan 'none', manh cuoi giu kind goc."""
    units = []
    for text, kind in segs:
        if len(text) <= max_chars:
            units.append((text, kind))
            continue
        words = text.split()
        piece, pieces = "", []
        for w in words:
            if len(piece) + len(w) + 1 <= max_chars:
                piece = (piece + " " + w).strip()
            else:
                pieces.append(piece)
                piece = w
        if piece:
            pieces.append(piece)
        for i, p in enumerate(pieces):
            units.append((p, kind if i == len(pieces) - 1 else "none"))
    return units


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
    ap.add_argument("--tts-ckpt", default=f"{TAC_RUN}/best_model.pth")
    ap.add_argument("--tts-config", default=f"{TAC_RUN}/config.json")
    ap.add_argument("--voc-ckpt", default=f"{HIFI_RUN}/checkpoint_68572.pth")
    ap.add_argument("--voc-config", default=f"{HIFI_RUN}/config.json")
    ap.add_argument("--out-dir", default="demo/tacotron_hifigan")
    ap.add_argument("--sentences", default="demo/sentences.txt",
                    help="File text chua cau demo (moi dong 1 cau)")
    ap.add_argument("--text", default=None,
                    help="CHE DO VAN BAN DAI: doc 1 file .txt -> 1 wav (thay vi demo1-5)")
    ap.add_argument("--out", default="demo/tacotron_hifigan/bai_dai.wav",
                    help="File wav dau ra cho che do --text")
    ap.add_argument("--max-chars", type=int, default=200,
                    help="Gioi han ky tu moi don vi tong hop (che do --text)")
    ap.add_argument("--no-pause", action="store_true",
                    help="Tong hop ca cau 1 lan (khong chen khoang lang)")
    ap.add_argument("--sil-short", type=float, default=0.2, help="Nghi dau phay , ; : (s)")
    ap.add_argument("--sil-long", type=float, default=0.3, help="Nghi dau cham . ! ? (s)")
    ap.add_argument("--sil-none", type=float, default=0.1, help="Nghi mac dinh giua doan (s)")
    ap.add_argument("--trim-thresh", type=float, default=0.015,
                    help="Nguong cat lang dau/cuoi moi doan (0=tat)")
    args = ap.parse_args()

    for p in (args.tts_ckpt, args.tts_config, args.voc_ckpt, args.voc_config, args.sentences):
        if not Path(p).exists():
            raise SystemExit(f"[error] Khong tim thay: {p}")

    sentences = load_sentences(args.sentences)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from TTS.utils.synthesizer import Synthesizer
    import soundfile as sf

    print(f"[load] Tacotron2: {args.tts_ckpt}")
    print(f"[load] HiFi-GAN : {args.voc_ckpt}")
    synth = Synthesizer(
        tts_checkpoint=args.tts_ckpt,
        tts_config_path=args.tts_config,
        vocoder_checkpoint=args.voc_ckpt,
        vocoder_config=args.voc_config,
        use_cuda=torch.cuda.is_available(),
    )
    sr = synth.output_sample_rate
    # Chan over-generate: Tacotron2 khong dung dut khoat -> cap so buoc decoder
    # (1 doan <=200 ky tu ~ <210 buoc @ r=3; 400 du headroom + nhanh hon ~16x).
    try:
        synth.tts_model.decoder.max_decoder_steps = 400
        print("[info] max_decoder_steps = 400 (chan over-generate)")
    except Exception as e:
        print(f"[warn] khong set duoc max_decoder_steps: {e}")
    print(f"[info] {len(sentences)} cau | {sr} Hz | Ngat nghi: {'TAT' if args.no_pause else 'BAT'}\n")

    def trim_sil(wav, pad=0.02):
        if args.trim_thresh <= 0 or len(wav) == 0:
            return wav
        idx = np.where(np.abs(wav) > args.trim_thresh)[0]
        if len(idx) == 0:
            return wav
        p = int(pad * sr)
        return wav[max(0, idx[0] - p):min(len(wav), idx[-1] + p)]

    def trim_tail(wav, frac=0.22, win_ms=30, pad_ms=60):
        """Cat duoi 'ummm' do Tacotron2 over-generate: bo phan duoi co nang luong
        < frac*dinh (tieng lam bam luon yeu hon giong that)."""
        if len(wav) == 0:
            return wav
        win = max(1, int(win_ms / 1000 * sr))
        n = len(wav) // win
        if n < 2:
            return wav
        env = np.array([np.sqrt((wav[i * win:(i + 1) * win] ** 2).mean()) for i in range(n)])
        mx = env.max()
        if mx <= 0:
            return wav
        strong = np.where(env > frac * mx)[0]
        if len(strong) == 0:
            return wav
        end = min(len(wav), (strong[-1] + 1) * win + int(pad_ms / 1000 * sr))
        return wav[:end]

    def apply_fade(wav, fade_ms=10):
        n = min(int(fade_ms / 1000 * sr), len(wav) // 2)
        if n > 0:
            wav = wav.copy()
            wav[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
            wav[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
        return wav

    def synth_seg(text):
        """Tong hop 1 doan: strip dau cau (Tacotron2 khong co vocab dau cau)."""
        text = strip_punct(text).strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        wav = synth.tts(text, split_sentences=False)
        wav = np.asarray(wav, dtype=np.float32)
        return apply_fade(trim_tail(trim_sil(wav)))

    SIL = {"long": args.sil_long, "short": args.sil_short, "none": args.sil_none}

    # ---- CHE DO VAN BAN DAI: doc 1 file -> chunk -> 1 wav ----
    if args.text:
        text_in = args.text
        if Path(text_in).exists():
            text_in = Path(text_in).read_text(encoding="utf-8")
        norm = normalize(text_in)
        units = make_units(split_for_pause(norm), args.max_chars)
        print(f"[long] {len(norm)} ky tu, ~{len(norm.split())} tu -> {len(units)} don vi (<= {args.max_chars} ky tu)")
        pieces, t0 = [], time.time()
        for k, (u, kind) in enumerate(units):
            p = synth_seg(u)
            if len(p):
                pieces.append(p)
                if k < len(units) - 1:
                    pieces.append(np.zeros(int(SIL[kind] * sr), dtype=np.float32))
            if (k + 1) % 10 == 0:
                print(f"  ... {k + 1}/{len(units)} don vi")
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.out, audio, sr)
        dur, el = len(audio) / sr, time.time() - t0
        print(f"\n[done] {args.out}")
        print(f"       Audio: {dur:.1f}s | xu ly: {el:.1f}s | RTF = {el / dur if dur else 0:.3f}")
        return

    total_audio, total_wall = 0.0, 0.0
    for i, raw in enumerate(sentences, 1):
        norm = normalize(raw)
        out_wav = out_dir / f"demo{i}.wav"
        t0 = time.time()

        if args.no_pause:
            audio = synth_seg(norm)
        else:
            segs = split_for_pause(norm)
            chunks = []
            for j, (seg, kind) in enumerate(segs):
                piece = synth_seg(seg)
                if len(piece):
                    chunks.append(piece)
                    if j < len(segs) - 1:
                        chunks.append(np.zeros(int(SIL[kind] * sr), dtype=np.float32))
            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

        sf.write(str(out_wav), audio, sr)
        elapsed = time.time() - t0
        dur = len(audio) / sr
        total_audio += dur
        total_wall += elapsed
        print(f"[{i}] {dur:4.1f}s -> {out_wav}")
        print(f"    raw : {raw[:65]}")
        print(f"    norm: {norm[:65]}")

    rtf = total_wall / total_audio if total_audio else 0
    print(f"\n[done] {len(sentences)} file trong: {out_dir}")
    print(f"       Tong audio: {total_audio:.1f}s | xu ly: {total_wall:.1f}s | RTF = {rtf:.3f}")


if __name__ == "__main__":
    main()
