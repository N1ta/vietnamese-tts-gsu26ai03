#!/usr/bin/env python3
"""
demo_vixtts.py
==============
Sinh demo bang viXTTS (XTTS v2 fine-tune tieng Viet) — CLONE giong tu 1 clip mau.
KHONG can train: zero-shot voice cloning, 24kHz.

- Chuan hoa text dung chung vi_normalize (so/ngay/viet tat/toan).
- Patch cho phep ngon ngu 'vi' (code XTTS goc chan, nhung vocab da co tieng Viet).
- Che do --text: doc van ban dai (chunk).

Cach dung:
    python demo_vixtts.py --ref dataset/MEN_dataset/wavs/0006.wav --out-dir demo/vixtts_men
    python demo_vixtts.py --ref ... --text demo/bai_dai.txt --out demo/vixtts_men/bai_dai.wav
    python demo_vixtts.py --ref ... --temperature 0.5
"""
import os, sys, re, argparse, time
os.environ["COQUI_TOS_AGREED"] = "1"
from pathlib import Path
import numpy as np

from vi_normalize import normalize, split_for_pause

VIXTTS_DIR = "vixtts_model"
SR = 24000


def load_sentences(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def make_units(segs, max_chars=200):
    """Gom doan (tach o dau cau) thanh don vi <= max_chars."""
    units = []
    for text, kind in segs:
        if len(text) <= max_chars:
            units.append((text, kind)); continue
        words, piece, pieces = text.split(), "", []
        for w in words:
            if len(piece) + len(w) + 1 <= max_chars:
                piece = (piece + " " + w).strip()
            else:
                pieces.append(piece); piece = w
        if piece: pieces.append(piece)
        for i, p in enumerate(pieces):
            units.append((p, kind if i == len(pieces) - 1 else "none"))
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="dataset/MEN_dataset/wavs/0006.wav", help="Clip mau de clone giong")
    ap.add_argument("--out-dir", default="demo/vixtts_men")
    ap.add_argument("--sentences", default="demo/sentences.txt")
    ap.add_argument("--text", default=None, help="Che do van ban dai: 1 file -> 1 wav")
    ap.add_argument("--out", default="demo/vixtts_men/bai_dai.wav")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max-chars", type=int, default=200)
    ap.add_argument("--sil", type=float, default=0.25, help="Khoang lang giua chunk (s)")
    args = ap.parse_args()

    # --- patch 'vi' ---
    from TTS.tts.layers.xtts import tokenizer as xtok
    _orig = xtok.VoiceBpeTokenizer.preprocess_text
    def _p(self, txt, lang):
        if lang == "vi":
            return re.sub(r"\s+", " ", txt.lower()).strip()
        return _orig(self, txt, lang)
    xtok.VoiceBpeTokenizer.preprocess_text = _p
    xtok.VoiceBpeTokenizer.check_input_length = lambda self, txt, lang: None

    import torch, soundfile as sf
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    print(f"[load] viXTTS | ref: {args.ref}")
    config = XttsConfig(); config.load_json(os.path.join(VIXTTS_DIR, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=VIXTTS_DIR, use_deepspeed=False)
    model.cuda()
    gpt_cond, spk = model.get_conditioning_latents(audio_path=[args.ref], gpt_cond_len=6, max_ref_length=30)
    print(f"[info] temperature={args.temperature} | 24000 Hz\n")

    def synth(text):
        text = normalize(text)  # GIU dau cau (viXTTS dung cho ngu dieu)
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        out = model.inference(text, "vi", gpt_cond, spk,
                              temperature=args.temperature, enable_text_splitting=False)
        return np.asarray(out["wav"], dtype=np.float32)

    # ---- che do van ban dai ----
    if args.text:
        text_in = args.text
        if Path(text_in).exists():
            text_in = Path(text_in).read_text(encoding="utf-8")
        units = make_units(split_for_pause(normalize(text_in)), args.max_chars)
        print(f"[long] {len(units)} don vi tong hop")
        pieces, t0 = [], time.time()
        for k, (u, _) in enumerate(units):
            out = model.inference(u, "vi", gpt_cond, spk, temperature=args.temperature, enable_text_splitting=False)
            pieces.append(np.asarray(out["wav"], dtype=np.float32))
            pieces.append(np.zeros(int(args.sil * SR), dtype=np.float32))
            if (k + 1) % 10 == 0: print(f"  ... {k+1}/{len(units)}")
        audio = np.concatenate(pieces)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.out, audio, SR)
        print(f"\n[done] {args.out} | {len(audio)/SR:.1f}s | xu ly {time.time()-t0:.1f}s")
        return

    # ---- che do 5 cau ----
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sents = load_sentences(args.sentences)
    t0 = time.time()
    for i, raw in enumerate(sents, 1):
        wav = synth(raw)
        f = out_dir / f"demo{i}.wav"
        sf.write(str(f), wav, SR)
        print(f"[{i}] {len(wav)/SR:4.1f}s -> {f}")
        print(f"    {normalize(raw)[:70]}")
    print(f"\n[done] {len(sents)} file trong {out_dir} | {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
