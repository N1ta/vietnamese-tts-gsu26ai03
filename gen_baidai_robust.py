#!/usr/bin/env python3
"""
gen_baidai_robust.py — Sinh BAI DAI "sach" bang F5 (chong artifact). CHAY f5_env.

  1. Cat doan NHO theo dau cau (split_for_pause + make_units, BO dau cau) ->
     doan ngan F5 doc chuan, it glitch (giong "ban cu" nghe hay hon coarse-chunk).
  2. Best-of-K moi doan: sinh K lan, chon ban do dai TRUNG VI -> loai ban bi
     phot them am/lap tu (deu LAM DAI hon).
  3. Chen khoang lang theo loai ngat (long/short/none).
  4. Co bao doan nao sau best-of-K van dai bat thuong (> 1.5x du kien).

Dung:
    python gen_baidai_robust.py --ref dataset/WOMEN_dataset/wavs/0022.wav \
        --ref-text "..." --out demo/f5_women/bai_dai.wav --speed 0.85 --nfe 48 --k 3
"""
import sys, os, argparse, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.getcwd())
import numpy as np
from vi_normalize import normalize, split_for_pause

MODEL_DIR = "f5_vi_model"
SIL = {"long": 0.30, "short": 0.20, "none": 0.10}


def make_units(segs, maxlen):
    """Cat doan qua dai o ranh gioi tu, giu loai ngat o manh cuoi."""
    units = []
    for text, kind in segs:
        if len(text) <= maxlen:
            units.append((text, kind)); continue
        words, piece, pieces = text.split(), "", []
        for w in words:
            if len(piece) + len(w) + 1 <= maxlen:
                piece = (piece + " " + w).strip()
            else:
                pieces.append(piece); piece = w
        if piece:
            pieces.append(piece)
        for i, p in enumerate(pieces):
            units.append((p, kind if i == len(pieces) - 1 else "none"))
    return units


def merge_short(units, minlen=28, maxlen=200):
    """GOP doan qua ngan (< minlen) voi hang xom -> F5 du ngu canh, khong nuot/
    slur chu (vd 'nghe sách nói', 'ngã và nặng'). Van tach o cuoi cau (long)."""
    out, buf, bk = [], "", "none"
    for text, kind in units:
        if not buf:
            buf, bk = text, kind
        elif len(buf) + 1 + len(text) <= maxlen:
            buf, bk = buf + " " + text, kind
        else:
            out.append((buf, "short")); buf, bk = text, kind
        if len(buf) >= minlen or kind == "long":
            out.append((buf, bk)); buf, bk = "", "none"
    if buf:
        out.append((buf, bk))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--ref-text-file", default=None,
                    help="Doc ref-text tu file utf-8 (tranh loi font khi goi tu .bat)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--long-text", default="demo/bai_dai.txt")
    ap.add_argument("--speed", type=float, default=0.85)
    ap.add_argument("--nfe", type=int, default=48)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--maxlen", type=int, default=200)
    args = ap.parse_args()

    ref_text = args.ref_text
    if args.ref_text_file:
        ref_text = open(args.ref_text_file, encoding="utf-8").read().strip()
    if not ref_text:
        ap.error("Can --ref-text hoac --ref-text-file")

    import soundfile as sf
    from f5_tts.api import F5TTS
    print(f"[load] F5 | ref {args.ref} | k={args.k} nfe={args.nfe} speed={args.speed}")
    f5 = F5TTS(model="F5TTS_Base",
               ckpt_file=os.path.join(MODEL_DIR, "model_last.pt"),
               vocab_file=os.path.join(MODEL_DIR, "config.json"))

    text = open(args.long_text, encoding="utf-8").read()
    units = make_units(split_for_pause(normalize(text)), args.maxlen)
    units = merge_short(units, minlen=28, maxlen=args.maxlen)  # gop doan qua ngan
    print(f"[chunks] {len(units)} doan (da gop ngan, cat theo dau cau, best-of-{args.k})")

    SR = 24000
    pieces, flags = [], []
    t0 = time.time()
    for idx, (ch, kind) in enumerate(units, 1):
        n_syll = max(1, len(ch.split()))
        exp = n_syll * 0.20 / args.speed
        cands, sr = [], SR
        for _ in range(args.k):
            wav, sr, _ = f5.infer(ref_file=args.ref, ref_text=ref_text,
                                  gen_text=ch, remove_silence=True,
                                  speed=args.speed, nfe_step=args.nfe)
            cands.append(np.asarray(wav, dtype=np.float32))
        durs = [len(w) / sr for w in cands]
        order = sorted(range(len(cands)), key=lambda i: durs[i])
        pick = order[len(order) // 2]                 # TRUNG VI
        wav = cands[pick]
        wav = wav / (np.abs(wav).max() + 1e-9) * 0.95
        ratio = durs[pick] / exp
        if ratio > 1.5:
            flags.append((idx, ch[:45], round(ratio, 2)))
        if idx % 20 == 0 or ratio > 1.5:
            print(f"[{idx:3d}/{len(units)}] {durs[pick]:.1f}s x{ratio:.2f}"
                  f"{'  <== soi lai' if ratio>1.5 else ''}")
        pieces.append(wav)
        pieces.append(np.zeros(int(SIL.get(kind, 0.12) * sr), dtype=np.float32))

    audio = np.concatenate(pieces)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sf.write(args.out, audio, SR)
    print(f"\n[done] {args.out} | {len(audio)/SR:.0f}s | {time.time()-t0:.0f}s")
    if flags:
        print(f"[!] {len(flags)} doan dai bat thuong (soi lai):")
        for i, txt, r in flags:
            print(f"    doan {i} (x{r}): {txt}...")
    else:
        print("[ok] Khong doan nao dai bat thuong.")


if __name__ == "__main__":
    main()
