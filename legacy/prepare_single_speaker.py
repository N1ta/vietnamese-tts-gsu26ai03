#!/usr/bin/env python3
"""
prepare_single_speaker.py
=========================
Select the SINGLE VIVOS speaker with the largest total audio duration,
then emit metadata files compatible with Tacotron2, FastSpeech2 and VITS.

Project layout (matches the screenshot GSU26AI03 - SU26):

    dataset/
      vivos/
        train/
          waves/<SPK>/<SPK>_<id>.wav
          prompts.txt          # "<SPK>_<id> transcript ..."
          genders.txt
        test/
          waves/...
          prompts.txt
      wavs/                     # (output) flattened wavs for the chosen speaker
      test.txt
    models/
      tacotron2/
      FastSpeech2_vi/
      vits/

Run from the project root (the folder that contains dataset/ and models/).

    python prepare_single_speaker.py --vivos dataset/vivos --out dataset

Outputs (under <out>/single_speaker/<SPK>/):
    metadata_ljspeech.csv   ->  tacotron2 + VITS   (LJSpeech "wav|text" / "wav|text|spk")
    metadata_fastspeech2.txt -> FastSpeech2_vi     ("name|speaker|phonemes-or-text|raw")
    filelist_train.txt / filelist_val.txt
    wavs/  (symlinks or copies of the chosen speaker's wavs)
    stats.json (chosen speaker, totals, split sizes)
"""

import argparse
import json
import os
import random
import shutil
import wave
from collections import defaultdict
from pathlib import Path


# --------------------------------------------------------------------------- #
# Duration helpers
# --------------------------------------------------------------------------- #
def wav_duration_seconds(path: Path) -> float:
    """Read duration from a PCM WAV header (no external deps)."""
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return frames / float(rate) if rate else 0.0
    except Exception:
        # Fallback to soundfile/librosa if the file is not plain PCM WAV.
        try:
            import soundfile as sf  # type: ignore
            info = sf.info(str(path))
            return info.frames / float(info.samplerate)
        except Exception:
            return 0.0


# --------------------------------------------------------------------------- #
# VIVOS parsing
# --------------------------------------------------------------------------- #
def parse_prompts(prompts_file: Path):
    """
    VIVOS prompts.txt lines look like:
        VIVOSSPK01_R001 ALO XIN CHAO
    Returns dict: utt_id -> transcript
    """
    mapping = {}
    if not prompts_file.exists():
        return mapping
    with open(prompts_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
            elif len(parts) == 1:
                mapping[parts[0]] = ""
    return mapping


def find_wav(waves_root: Path, utt_id: str):
    """
    Locate the wav for an utterance id. VIVOS stores them as
        waves/<SPK>/<utt_id>.wav   where SPK is the prefix before '_'.
    """
    spk = utt_id.split("_")[0]
    cand = waves_root / spk / f"{utt_id}.wav"
    if cand.exists():
        return cand
    # Fallback: search anywhere under waves_root.
    hits = list(waves_root.rglob(f"{utt_id}.wav"))
    return hits[0] if hits else None


def collect_utterances(vivos_root: Path):
    """
    Scan train/ and test/ splits. Returns list of dicts:
        {utt_id, speaker, wav_path, text, duration}
    """
    records = []
    for split in ("train", "test"):
        split_dir = vivos_root / split
        if not split_dir.exists():
            continue
        prompts = parse_prompts(split_dir / "prompts.txt")
        waves_root = split_dir / "waves"
        if not waves_root.exists():
            # Some mirrors flatten to <split>/<SPK>/*.wav
            waves_root = split_dir
        for utt_id, text in prompts.items():
            wav = find_wav(waves_root, utt_id)
            if wav is None:
                continue
            spk = utt_id.split("_")[0]
            records.append(
                {
                    "utt_id": utt_id,
                    "speaker": spk,
                    "wav_path": wav,
                    "text": text,
                    "duration": wav_duration_seconds(wav),
                }
            )
    return records


# --------------------------------------------------------------------------- #
# Speaker selection
# --------------------------------------------------------------------------- #
def pick_longest_speaker(records):
    totals = defaultdict(float)
    counts = defaultdict(int)
    for r in records:
        totals[r["speaker"]] += r["duration"]
        counts[r["speaker"]] += 1
    if not totals:
        raise SystemExit("No utterances found. Check the --vivos path.")
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    best_spk, best_dur = ranked[0]
    return best_spk, best_dur, counts[best_spk], ranked


# --------------------------------------------------------------------------- #
# Metadata writers
# --------------------------------------------------------------------------- #
def write_metadata(records, speaker, out_root: Path, link_wavs: bool, val_ratio: float, seed: int):
    spk_records = [r for r in records if r["speaker"] == speaker]
    spk_records.sort(key=lambda r: r["utt_id"])

    dst_dir = out_root / "single_speaker" / speaker
    wav_dir = dst_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Materialise wavs (symlink by default, copy if requested) with flat names.
    for r in spk_records:
        target = wav_dir / f"{r['utt_id']}.wav"
        r["flat_wav"] = target
        if target.exists() or target.is_symlink():
            continue
        if link_wavs:
            try:
                os.symlink(os.path.abspath(r["wav_path"]), target)
            except (OSError, NotImplementedError):
                shutil.copy2(r["wav_path"], target)
        else:
            shutil.copy2(r["wav_path"], target)

    # --- Tacotron2 + VITS : LJSpeech-style "wav_basename|text" ----------------
    # Tacotron2's ljspeech loader expects "wav|text"; VITS multi/single expects
    # "wav_path|text" (single) or "wav_path|spk|text" (multi). Single speaker
    # here, so the 2-field form serves both.
    lj_path = dst_dir / "metadata_ljspeech.csv"
    with open(lj_path, "w", encoding="utf-8") as f:
        for r in spk_records:
            f.write(f"wavs/{r['utt_id']}.wav|{r['text']}\n")

    # VITS sometimes wants absolute paths in its filelists; provide that too.
    vits_path = dst_dir / "metadata_vits.txt"
    with open(vits_path, "w", encoding="utf-8") as f:
        for r in spk_records:
            f.write(f"{os.path.abspath(r['flat_wav'])}|{r['text']}\n")

    # --- FastSpeech2 : "basename|speaker|text|raw_text" -----------------------
    # FastSpeech2 (ming024 layout) reads train.txt/val.txt with 4 pipe fields.
    # We put raw text in both text slots; your G2P/MFA step fills phonemes later.
    fs2_path = dst_dir / "metadata_fastspeech2.txt"
    with open(fs2_path, "w", encoding="utf-8") as f:
        for r in spk_records:
            f.write(f"{r['utt_id']}|{speaker}|{{{r['text']}}}|{r['text']}\n")

    # --- Train / val split (shared) ------------------------------------------
    rng = random.Random(seed)
    shuffled = spk_records[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val, train = shuffled[:n_val], shuffled[n_val:]

    def dump(split_records, name, fmt):
        p = dst_dir / name
        with open(p, "w", encoding="utf-8") as f:
            for r in split_records:
                if fmt == "lj":
                    f.write(f"wavs/{r['utt_id']}.wav|{r['text']}\n")
                else:  # fs2
                    f.write(f"{r['utt_id']}|{speaker}|{{{r['text']}}}|{r['text']}\n")
        return p

    dump(train, "filelist_train.txt", "lj")
    dump(val, "filelist_val.txt", "lj")
    dump(train, "train_fastspeech2.txt", "fs2")
    dump(val, "val_fastspeech2.txt", "fs2")

    stats = {
        "speaker": speaker,
        "num_utterances": len(spk_records),
        "total_duration_sec": round(sum(r["duration"] for r in spk_records), 2),
        "total_duration_min": round(sum(r["duration"] for r in spk_records) / 60.0, 2),
        "train_size": len(train),
        "val_size": len(val),
        "outputs": {
            "tacotron2_vits_ljspeech": str(lj_path),
            "vits_abs": str(vits_path),
            "fastspeech2": str(fs2_path),
        },
    }
    with open(dst_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats, dst_dir


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Single-speaker VIVOS extractor.")
    ap.add_argument("--vivos", default="dataset/vivos", help="Path to VIVOS root.")
    ap.add_argument("--out", default="dataset", help="Output root.")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--copy", action="store_true",
                    help="Copy wavs instead of symlinking.")
    args = ap.parse_args()

    vivos_root = Path(args.vivos)
    out_root = Path(args.out)

    print(f"[scan] reading VIVOS from {vivos_root} ...")
    records = collect_utterances(vivos_root)
    print(f"[scan] found {len(records)} utterances "
          f"across {len(set(r['speaker'] for r in records))} speakers")

    speaker, dur, count, ranked = pick_longest_speaker(records)
    print(f"\n[select] longest speaker = {speaker} "
          f"({dur/60:.1f} min, {count} clips)")
    print("[select] top 5 speakers by duration:")
    for spk, d in ranked[:5]:
        print(f"         {spk:>14s}  {d/60:7.1f} min")

    stats, dst = write_metadata(
        records, speaker, out_root,
        link_wavs=not args.copy, val_ratio=args.val_ratio, seed=args.seed,
    )
    print(f"\n[done] wrote metadata + wavs to: {dst}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
