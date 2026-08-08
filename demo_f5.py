#!/usr/bin/env python3
"""
demo_f5.py — Sinh demo bang F5-TTS Vietnamese (clone giong tu 1 clip mau).
CHAY TRONG ENV RIENG f5_env (xung dot transformers voi coqui):
    C:\\Users\\Nita\\miniconda3\\envs\\f5_env\\python.exe demo_f5.py ...

- Chuan hoa text dung chung vi_normalize (so/ngay/viet tat).
- F5 xuat audio to -> chuan hoa peak 0.95 TRUOC khi luu (tranh clip).
- --with-long: sinh them bai dai (demo/bai_dai.txt).

Vi du:
    python demo_f5.py --ref dataset/MEN_dataset/wavs/0006.wav --ref-text "..." --out-dir demo/f5_men --with-long
"""
import os, sys, argparse, time, re
sys.path.insert(0, os.getcwd())
import numpy as np
from vi_normalize import normalize, split_for_pause

MODEL_DIR = "f5_vi_model"


def load_sentences(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def make_units(segs, max_chars):
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
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sentences", default="demo/sentences.txt")
    ap.add_argument("--with-long", action="store_true", help="Sinh them bai dai")
    ap.add_argument("--long-text", default="demo/bai_dai.txt")
    ap.add_argument("--max-chars", type=int, default=250)
    ap.add_argument("--sil", type=float, default=0.2)
    ap.add_argument("--speed", type=float, default=1.0, help="Toc do noi (<1 = cham hon)")
    ap.add_argument("--nfe", type=int, default=32, help="So buoc sinh (cao hon = ro/muot hon, cham hon)")
    ap.add_argument("--only", default=None,
                    help="Chi sinh lai cac cau chi dinh (vd '2' hoac '1,3'); rong = bo qua 5 cau, chi lam bai dai")
    args = ap.parse_args()

    import torch, soundfile as sf
    from f5_tts.api import F5TTS
    print(f"[load] F5-TTS Vietnamese | ref: {args.ref}")
    f5 = F5TTS(model="F5TTS_Base",
               ckpt_file=os.path.join(MODEL_DIR, "model_last.pt"),
               vocab_file=os.path.join(MODEL_DIR, "config.json"))
    SR = [None]

    def synth(gen_text):
        gen_text = normalize(gen_text)
        if not gen_text.strip():
            return np.zeros(0, dtype=np.float32)
        wav, sr, _ = f5.infer(ref_file=args.ref, ref_text=args.ref_text,
                              gen_text=gen_text, remove_silence=True,
                              speed=args.speed, nfe_step=args.nfe)
        SR[0] = sr
        wav = np.asarray(wav, dtype=np.float32)
        return wav / (np.abs(wav).max() + 1e-9) * 0.95   # chuan hoa TRUOC khi luu

    os.makedirs(args.out_dir, exist_ok=True)

    # --- 5 cau (co the loc bang --only de sinh lai rieng cau bi loi) ---
    only = None
    if args.only is not None:
        only = {int(x) for x in args.only.split(",") if x.strip().isdigit()}
    sents = load_sentences(args.sentences)
    t0 = time.time()
    for i, raw in enumerate(sents, 1):
        if only is not None and i not in only:
            continue
        wav = synth(raw)
        sf.write(os.path.join(args.out_dir, f"demo{i}.wav"), wav, SR[0])
        print(f"[{i}] {len(wav)/SR[0]:4.1f}s | {normalize(raw)[:60]}")
    print(f"[done] 5 cau -> {args.out_dir} | {time.time()-t0:.0f}s")

    # --- bai dai ---
    if args.with_long:
        text = open(args.long_text, encoding="utf-8").read()
        units = make_units(split_for_pause(normalize(text)), args.max_chars)
        print(f"[long] {len(units)} don vi...")
        pieces, t1 = [], time.time()
        for k, (u, _) in enumerate(units):
            wav, sr, _ = f5.infer(ref_file=args.ref, ref_text=args.ref_text, gen_text=u, remove_silence=True,
                                  speed=args.speed, nfe_step=args.nfe)
            wav = np.asarray(wav, dtype=np.float32); wav = wav/(np.abs(wav).max()+1e-9)*0.95
            pieces.append(wav); pieces.append(np.zeros(int(args.sil*sr), dtype=np.float32))
            if (k+1) % 10 == 0: print(f"  ... {k+1}/{len(units)}")
        audio = np.concatenate(pieces)
        sf.write(os.path.join(args.out_dir, "bai_dai.wav"), audio, sr)
        print(f"[done] bai_dai {len(audio)/sr:.0f}s | {time.time()-t1:.0f}s")


if __name__ == "__main__":
    main()
