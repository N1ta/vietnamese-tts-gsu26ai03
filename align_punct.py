#!/usr/bin/env python3
"""
align_punct.py  (D3 - khoi phuc dau cau bang FORCED ALIGNMENT)
==============================================================
Can transcript goc vao audio bang wav2vec2 tieng Viet -> biet thoi diem moi tu
-> phat hien khoang lang giua 2 tu (>= MIN_PAUSE) -> chen DAU PHAY dung ranh gioi
tu; cuoi cau them DAU CHAM. GIU NGUYEN chu goc.

Ket qua: dataset/infore_coqui/metadata_punct.csv  (id|raw|text_co_dau_cau)
-> dung de train lai cho model hoc NGAT NGHI tu nhien.

Cach dung:
    python align_punct.py --demo 8                 # xem 8 file (khong ghi)
    python align_punct.py --out dataset/infore_coqui/metadata_punct.csv   # chay toan bo
    python align_punct.py --out ... --min-pause 0.3
"""

import argparse
import os
import unicodedata
from pathlib import Path

import torch
import soundfile as sf
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *a, **k: None  # model .bin tin cay
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torchaudio.functional import forced_align, merge_tokens

DATA_DIR = r"dataset/infore_16k_denoised_2"
MODEL = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"


def load_model(device):
    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval().to(device)
    return proc, model


def punctuate(fid, proc, model, vocab, blank, device, min_pause):
    wav, sr = sf.read(os.path.join(DATA_DIR, fid + ".wav"))
    raw = Path(os.path.join(DATA_DIR, fid + ".txt")).read_text(encoding="utf-8").strip()
    text = unicodedata.normalize("NFC", raw.lower())
    words = text.split()
    if not words:
        return raw, ""

    iv = proc(wav, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        log_probs = torch.log_softmax(model(iv).logits, dim=-1)
    n_frames = log_probs.shape[1]
    ratio = len(wav) / n_frames / sr

    target, word_ranges = [], []
    for w in words:
        a = len(target)
        for ch in w:
            if ch in vocab:
                target.append(vocab[ch])
        word_ranges.append((a, len(target)))
    if not target:
        return raw, text + "."

    targets = torch.tensor([target], dtype=torch.int32, device=device)
    aligned, scores = forced_align(log_probs, targets, blank=blank)
    spans = [s for s in merge_tokens(aligned[0], scores[0]) if s.token != blank]
    if len(spans) != len(target):  # alignment loi -> chi them dau cham
        return raw, text + "."

    wt = []
    for (a, b) in word_ranges:
        wt.append((spans[a].start * ratio, spans[b-1].end * ratio) if a < b else None)

    out = []
    for i, w in enumerate(words):
        token = w
        if i < len(words) - 1 and wt[i] and wt[i+1]:
            if wt[i+1][0] - wt[i][1] >= min_pause:
                token = w + ","
        out.append(token)
    return raw, " ".join(out) + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--min-pause", type=float, default=0.25)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}, min_pause={args.min_pause}")
    proc, model = load_model(device)
    vocab = proc.tokenizer.get_vocab()
    blank = proc.tokenizer.pad_token_id

    all_ids = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".wav"))

    if args.demo:
        ids = all_ids[::max(1, len(all_ids)//args.demo)][:args.demo]
        for fid in ids:
            raw, punct = punctuate(fid, proc, model, vocab, blank, device, args.min_pause)
            print(f"[{fid}] {punct}")
        return

    if not args.out:
        ap.error("Chon --demo N hoac --out file")

    lines, errs = [], 0
    for k, fid in enumerate(all_ids):
        try:
            raw, punct = punctuate(fid, proc, model, vocab, blank, device, args.min_pause)
            lines.append(f"{fid}|{raw}|{punct}")
        except Exception as e:
            errs += 1
            if errs <= 5:
                print(f"  [loi] {fid}: {e}")
        if (k + 1) % 500 == 0:
            print(f"  ...{k+1}/{len(all_ids)}  (loi: {errs})")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    # thong ke so dau phay
    n_comma = sum(l.count(",") for l in lines)
    print(f"\n[done] Ghi {len(lines)} dong -> {args.out}")
    print(f"       Loi: {errs} | Tong dau phay chen: {n_comma} (TB {n_comma/max(len(lines),1):.2f}/cau)")


if __name__ == "__main__":
    main()
