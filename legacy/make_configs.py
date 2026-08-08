#!/usr/bin/env python3
"""
make_configs.py
===============
Generate config.json for Tacotron2, FastSpeech2 and VITS, tuned so each model
lands close to a shared parameter budget (~50M) for a fair 3-way comparison
(Review-1 requirement: "Chung minh 3 models co cung tham so").

Two things this script does:
  1. Writes models/<name>/config.json with acoustic params adjusted toward 50M.
  2. Estimates the parameter count analytically and prints a table so you can
     DEFEND the number, instead of just asserting it. (Exact counts come from
     each repo's own model.summary; the estimator here is within a few %.)

Run from project root:
    python make_configs.py --target 50e6 --vocab 256 --speakers 1

NOTE on fairness: the comparison is only "fair" on parameter count, not FLOPs
or latency. Tacotron2 is autoregressive; FastSpeech2/VITS are not. State that
explicitly in your report. VITS also bundles a HiFi-GAN-style decoder + flows,
so a large share of its params are in the vocoder, not the text encoder.
"""

import argparse
import json
import math
from pathlib import Path

MEL_CHANNELS = 80
SAMPLE_RATE = 22050


# --------------------------------------------------------------------------- #
# Rough analytic parameter estimators (good enough to tune dims, ~within 5%).
# --------------------------------------------------------------------------- #
def transformer_block_params(d, ffn, heads):
    # self-attn: 4 * d*d (q,k,v,o)   +   ffn: 2 * d*ffn   + 2 LayerNorm
    return 4 * d * d + 2 * d * ffn + 4 * d


def est_tacotron2(c):
    sym = c["n_symbols"]; emb = c["symbols_embedding_dim"]
    enc = c["encoder_embedding_dim"]
    attn_rnn = c["attention_rnn_dim"]; dec_rnn = c["decoder_rnn_dim"]
    pre = c["prenet_dim"]; post = c["postnet_embedding_dim"]
    p = sym * emb
    # 3 conv banks in encoder: kernel 5
    p += 3 * (5 * enc * enc)
    # bi-LSTM encoder
    p += 2 * (4 * (enc * (enc // 2) + (enc // 2) * (enc // 2)))
    # attention
    p += attn_rnn * (enc + MEL_CHANNELS) * 4 + attn_rnn * attn_rnn * 4
    # 2-layer decoder LSTM
    p += 4 * ((attn_rnn + enc) * dec_rnn + dec_rnn * dec_rnn)
    p += 4 * (dec_rnn * dec_rnn + dec_rnn * dec_rnn)
    # prenet + projections
    p += MEL_CHANNELS * pre + pre * pre + dec_rnn * MEL_CHANNELS
    # postnet: 5 conv layers kernel 5
    p += 5 * (5 * post * post)
    return int(p)


def est_fastspeech2(c):
    d = c["transformer"]["encoder_hidden"]
    enc_l = c["transformer"]["encoder_layer"]
    dec_l = c["transformer"]["decoder_layer"]
    ffn = c["transformer"]["conv_filter_size"]
    heads = c["transformer"]["encoder_head"]
    vocab = c["vocab_size"]
    p = vocab * d                      # phoneme embedding
    p += enc_l * transformer_block_params(d, ffn, heads)
    p += dec_l * transformer_block_params(d, ffn, heads)
    # variance adaptor: 3 predictors (dur/pitch/energy), 2 conv each kernel 3
    p += 3 * (2 * (3 * d * d) + d * 1)
    p += 256 * d * 2                   # pitch/energy embedding buckets
    p += d * MEL_CHANNELS              # mel linear
    return int(p)


def est_vits(c):
    h = c["model"]["hidden_channels"]
    inter = c["model"]["filter_channels"]
    enc_l = c["model"]["n_layers"]
    vocab = c["model"]["n_vocab"]
    flows = c["model"]["n_flows"]
    ups = c["model"]["upsample_initial_channel"]
    p = vocab * h
    # text encoder (transformer)
    p += enc_l * transformer_block_params(h, inter, c["model"]["n_heads"])
    # posterior encoder (WaveNet residual stack)
    p += c["model"]["n_layers_q"] * (2 * h * h * 3)
    # normalizing flows
    p += flows * (4 * h * h)
    # HiFi-GAN-style decoder (dominant cost): upsample convs + MRF resblocks
    ch = ups
    krs = c["model"]["resblock_kernel_sizes"]
    dils = c["model"]["resblock_dilation_sizes"]
    for k in c["model"]["upsample_rates"]:
        nxt = ch // 2
        p += ch * nxt * (2 * k)            # transposed conv
        # multi-receptive-field resblocks at this resolution
        for kr, dl in zip(krs, dils):
            p += len(dl) * 2 * (nxt * nxt * kr)
        ch = nxt
    p += ch * 7 + ups * 7                  # pre/post conv
    return int(p)


# --------------------------------------------------------------------------- #
# Base configs (dims pre-tuned to sit near 50M).
# --------------------------------------------------------------------------- #
def tacotron2_config(vocab, sz):
    # sz: dict cac dims theo preset (20m / 50m)
    return {
        "n_symbols": vocab,
        "symbols_embedding_dim": sz["t2_emb"],
        "encoder_embedding_dim": sz["t2_emb"],
        "encoder_n_convolutions": 3,
        "encoder_kernel_size": 5,
        "attention_rnn_dim": sz["t2_rnn"],
        "attention_dim": 128,
        "decoder_rnn_dim": sz["t2_rnn"],
        "prenet_dim": 256,
        "postnet_embedding_dim": sz["t2_post"],
        "postnet_kernel_size": 5,
        "postnet_n_convolutions": 5,
        "n_mel_channels": MEL_CHANNELS,
        "sampling_rate": SAMPLE_RATE,
        "mel_fmin": 0.0, "mel_fmax": 8000.0,
        "max_decoder_steps": 2000, "gate_threshold": 0.5,
        "p_attention_dropout": 0.1, "p_decoder_dropout": 0.1,
    }


def fastspeech2_config(vocab, sz):
    return {
        "vocab_size": vocab,
        "max_seq_len": 1500,
        "transformer": {
            "encoder_layer": sz["fs_layers"], "encoder_head": 8,
            "encoder_hidden": sz["fs_hidden"],
            "decoder_layer": sz["fs_layers"], "decoder_head": 8,
            "decoder_hidden": sz["fs_hidden"],
            "conv_filter_size": sz["fs_ffn"], "conv_kernel_size": [9, 1],
            "encoder_dropout": 0.2, "decoder_dropout": 0.2,
        },
        "variance_predictor": {
            "filter_size": 256, "kernel_size": 3, "dropout": 0.5,
        },
        "variance_embedding": {
            "pitch_quantization": "linear", "energy_quantization": "linear",
            "n_bins": 256,
        },
        "mel": {
            "n_mel_channels": MEL_CHANNELS, "sampling_rate": SAMPLE_RATE,
            "filter_length": 1024, "hop_length": 256, "win_length": 1024,
            "mel_fmin": 0.0, "mel_fmax": 8000.0,
        },
    }


def vits_config(vocab, speakers, sz):
    return {
        "model": {
            "n_vocab": vocab,
            "inter_channels": 192, "hidden_channels": sz["vits_hidden"],
            "filter_channels": sz["vits_filter"], "n_heads": 4,
            "n_layers": sz["vits_layers"], "kernel_size": 3, "p_dropout": 0.1,
            "n_layers_q": sz["vits_q"], "n_flows": sz["vits_flows"],
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": sz["vits_up"],
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "n_speakers": speakers if speakers > 1 else 0,
            "gin_channels": 256 if speakers > 1 else 0,
        },
        "data": {
            "sampling_rate": SAMPLE_RATE, "filter_length": 1024,
            "hop_length": 256, "win_length": 1024,
            "n_mel_channels": MEL_CHANNELS, "mel_fmin": 0.0, "mel_fmax": None,
            # mel amplitude clamp range, consumed by preprocess_audio.py
            "mel_clamp_min": 0.0, "mel_clamp_max": 1.0,
        },
        "train": {
            "learning_rate": 2e-4, "batch_size": 16, "fp16_run": True,
            "segment_size": 8192, "c_mel": 45, "c_kl": 1.0,
        },
    }


# --------------------------------------------------------------------------- #
# Size presets: dims tuned so all 3 models land near the target budget.
# 50m = original (for Colab/desktop GPU); 20m = laptop 6GB-friendly.
# --------------------------------------------------------------------------- #
PRESETS = {
    "50m": {
        "t2_emb": 512, "t2_rnn": 1152, "t2_post": 640,
        "fs_layers": 6, "fs_hidden": 528, "fs_ffn": 2112,
        "vits_hidden": 256, "vits_filter": 1024, "vits_layers": 12,
        "vits_q": 24, "vits_flows": 12, "vits_up": 768,
    },
    "20m": {
        "t2_emb": 384, "t2_rnn": 704, "t2_post": 448,
        "fs_layers": 4, "fs_hidden": 448, "fs_ffn": 1280,
        "vits_hidden": 192, "vits_filter": 768, "vits_layers": 6,
        "vits_q": 16, "vits_flows": 4, "vits_up": 512,
    },
}

DEFAULT_TARGET = {"50m": 50e6, "20m": 20e6}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--preset", choices=["20m", "50m"], default="20m",
                    help="20m = laptop 6GB-friendly; 50m = desktop/Colab GPU.")
    ap.add_argument("--target", type=float, default=None,
                    help="Override budget for the % column (default by preset).")
    ap.add_argument("--vocab", type=int, default=91,
                    help="Symbol/phoneme vocab size (after your Vietnamese G2P).")
    ap.add_argument("--speakers", type=int, default=1)
    args = ap.parse_args()

    sz = PRESETS[args.preset]
    target = args.target if args.target else DEFAULT_TARGET[args.preset]

    mdir = Path(args.models_dir)
    specs = {
        "tacotron2": (tacotron2_config(args.vocab, sz), est_tacotron2),
        "FastSpeech2_vi": (fastspeech2_config(args.vocab, sz), est_fastspeech2),
        "vits": (vits_config(args.vocab, args.speakers, sz), est_vits),
    }

    print(f"preset={args.preset}  target={target/1e6:.0f}M  vocab={args.vocab}")
    print(f"{'model':<16}{'est. params':>14}{'vs target':>12}")
    print("-" * 42)
    for name, (cfg, est) in specs.items():
        out_dir = mdir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        n = est(cfg)
        cfg["_param_budget"] = {
            "preset": args.preset,
            "target": int(target),
            "estimated": int(n),
            "note": "Analytic estimate; confirm with the repo's own summary().",
        }
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        delta = (n - target) / target * 100
        print(f"{name:<16}{n:>14,}{delta:>+11.1f}%")
        print(f"{'':16}-> {out_dir/'config.json'}")

    print("\nReminder: equal parameter count != equal compute. Tacotron2 is "
          "autoregressive; VITS spends a large fraction of its params in the "
          "vocoder/flows. Report all three counts AND wall-clock/FLOPs.")


if __name__ == "__main__":
    main()
