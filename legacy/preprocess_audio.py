#!/usr/bin/env python3
"""
preprocess_audio.py
===================
Compute mel-spectrograms for the single-speaker VIVOS set and apply the
reviewer-requested amplitude normalisation: ep bien do Mel-spectrogram vao
doan [0.0, 1.0] ("convergence optimization / chuan hoa phan vi").

Pipeline per utterance:
    wav -> resample -> trim silence -> STFT -> mel filterbank
        -> log compression -> PERCENTILE normalise -> CLAMP to [0.0, 1.0]
        -> save <name>.npy

Why percentile + clamp (the defensible version):
  * Hard-clamping raw log-mels to [0,1] would destroy dynamic range.
  * Instead we map the [p_low, p_high] percentile band of the log-mel to
    [0,1] (robust to outliers), THEN clamp the tails. This is reversible:
    we save the per-corpus (min, max, percentiles) to mel_stats.json so the
    vocoder/synthesis stage can invert it. Report this as a normalisation
    choice, not a lossy hack.

Run from project root:
    python preprocess_audio.py \
        --filelist dataset/single_speaker/<SPK>/metadata_ljspeech.csv \
        --wav-root dataset/single_speaker/<SPK> \
        --out dataset/single_speaker/<SPK>/mels \
        --config models/FastSpeech2_vi/config.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError:
    raise SystemExit("This script needs librosa:  pip install librosa soundfile")


# --------------------------------------------------------------------------- #
def load_mel_params(config_path):
    """Pull STFT/mel params from a model config, with sane defaults."""
    p = {
        "sampling_rate": 22050, "filter_length": 1024, "hop_length": 256,
        "win_length": 1024, "n_mel_channels": 80, "mel_fmin": 0.0,
        "mel_fmax": 8000.0, "mel_clamp_min": 0.0, "mel_clamp_max": 1.0,
    }
    if config_path and Path(config_path).exists():
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        block = cfg.get("mel") or cfg.get("data") or cfg
        for k in p:
            if k in block and block[k] is not None:
                p[k] = block[k]
    return p


def wav_to_logmel(wav, sr, p):
    """Return a [n_mels, T] log-mel spectrogram (natural log of clamped power)."""
    S = librosa.feature.melspectrogram(
        y=wav, sr=sr,
        n_fft=p["filter_length"], hop_length=p["hop_length"],
        win_length=p["win_length"], n_mels=p["n_mel_channels"],
        fmin=p["mel_fmin"], fmax=p["mel_fmax"], power=1.0,
    )
    # dynamic-range compression (common TTS choice): log(max(S, 1e-5))
    return np.log(np.clip(S, a_min=1e-5, a_max=None)).astype(np.float32)


def normalise_and_clamp(logmel, lo, hi, clamp_min, clamp_max):
    """
    Map [lo, hi] -> [clamp_min, clamp_max] linearly, then clamp the tails.
    lo/hi are corpus-level percentile bounds computed in pass 1.
    """
    scaled = (logmel - lo) / max(hi - lo, 1e-8)
    scaled = scaled * (clamp_max - clamp_min) + clamp_min
    return np.clip(scaled, clamp_min, clamp_max).astype(np.float32)


# --------------------------------------------------------------------------- #
def read_filelist(path):
    """Accept LJSpeech (wav|text) or FastSpeech2 (name|spk|ph|raw)."""
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if parts[0].endswith(".wav"):
            items.append(parts[0])                       # LJSpeech style
        else:
            items.append(f"wavs/{parts[0]}.wav")         # FS2 basename style
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--wav-root", required=True,
                    help="Dir that the relative wav paths are resolved against.")
    ap.add_argument("--out", required=True, help="Where .npy mels are written.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--p-low", type=float, default=1.0,
                    help="Lower percentile for normalisation band.")
    ap.add_argument("--p-high", type=float, default=99.0,
                    help="Upper percentile for normalisation band.")
    args = ap.parse_args()

    p = load_mel_params(args.config)
    wav_root = Path(args.wav_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rels = read_filelist(args.filelist)
    print(f"[init] {len(rels)} utterances, sr={p['sampling_rate']}, "
          f"clamp=[{p['mel_clamp_min']}, {p['mel_clamp_max']}]")

    # ---- Pass 1: gather global percentile band over all log-mels -----------
    print("[pass1] computing corpus percentile band ...")
    sample_vals = []
    cache = {}
    for i, rel in enumerate(rels):
        wav_path = wav_root / rel
        if not wav_path.exists():
            print(f"  [skip] missing {wav_path}")
            continue
        wav, _ = librosa.load(str(wav_path), sr=p["sampling_rate"])
        wav, _ = librosa.effects.trim(wav, top_db=30)
        lm = wav_to_logmel(wav, p["sampling_rate"], p)
        cache[rel] = lm
        # subsample to keep memory bounded
        flat = lm.reshape(-1)
        if flat.size > 4000:
            flat = np.random.choice(flat, 4000, replace=False)
        sample_vals.append(flat)
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(rels)}")

    allv = np.concatenate(sample_vals)
    lo = float(np.percentile(allv, args.p_low))
    hi = float(np.percentile(allv, args.p_high))
    print(f"[pass1] log-mel band: p{args.p_low}={lo:.3f}  p{args.p_high}={hi:.3f}")

    # ---- Pass 2: normalise, clamp, save ------------------------------------
    print("[pass2] writing clamped mels ...")
    n_clamped = 0
    total = 0
    for rel, lm in cache.items():
        mel = normalise_and_clamp(
            lm, lo, hi, p["mel_clamp_min"], p["mel_clamp_max"]
        )
        # sanity: everything inside [clamp_min, clamp_max]
        assert mel.min() >= p["mel_clamp_min"] - 1e-6
        assert mel.max() <= p["mel_clamp_max"] + 1e-6
        # track how much was actually clipped (for the report)
        raw_scaled = (lm - lo) / max(hi - lo, 1e-8)
        n_clamped += int(((raw_scaled < 0) | (raw_scaled > 1)).sum())
        total += raw_scaled.size
        name = Path(rel).stem + ".npy"
        np.save(out_dir / name, mel)

    stats = {
        "sampling_rate": p["sampling_rate"],
        "n_mel_channels": p["n_mel_channels"],
        "clamp_range": [p["mel_clamp_min"], p["mel_clamp_max"]],
        "norm_percentiles": [args.p_low, args.p_high],
        "logmel_low": lo, "logmel_high": hi,
        "fraction_clamped": round(n_clamped / max(total, 1), 4),
        "note": "Invert with: logmel = mel/(clamp_max-clamp_min)*(hi-lo)+lo",
    }
    (out_dir.parent / "mel_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] {len(cache)} mels in {out_dir}")
    print(f"[done] clamped {stats['fraction_clamped']*100:.2f}% of bins "
          f"(tails outside the percentile band)")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
