#!/usr/bin/env python3
"""
train_coqui.py  (v3 - InFoRe 16kHz, Lua chon A: size mac dinh)
==============================================================
Sinh config.json DUNG SCHEMA COQUI cho 3 model (Tacotron2, FastPitch~FastSpeech2,
VITS) tren InFoRe single-speaker 16kHz, toi uu RTX 4050 6GB.

Dataset: dataset/infore_coqui/ (da tao bang prepare_infore.py)
  - 14835 cap wav/txt, ~25 gio, giong nu, 16000 Hz.
  - metadata.csv dinh dang LJSpeech 3 cot: id|raw|normalized

So sanh "cong bang": bao cao so tham so THAT cua ca 3 (Coqui in ra luc khoi tao,
vd "Model has XXXXX parameters") + toc do inference (RTF).

Cach dung (tu thu muc goc du an):
    python train_coqui.py --all
    python train_coqui.py --model vits

Sau do chay:
    python -m TTS.bin.train_tts --config_path models/<ten>/coqui_config.json
"""

import argparse
import json
from pathlib import Path

# Bo ky tu tieng Viet day du (chu thuong + dau). Khop voi text da normalize.
VI_CHARACTERS = ("abcdefghijklmnopqrstuvwxyz"
                 "àáâãèéêìíòóôõùúýăđĩũơư"
                 "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
VI_PUNCT = " "  # text da normalize chi con chu + khoang trang

# Tinh chinh VRAM 6GB cho tung model
GPU6 = {
    "tacotron2":     {"batch": 16, "eval_batch": 2, "fp16": True},
    "FastSpeech2_vi":{"batch": 16, "eval_batch": 2, "fp16": True},
    "vits":          {"batch": 6,  "eval_batch": 2, "fp16": True},
}


def characters_config(model_kind):
    # VITS dung lop ky tu rieng; Tacotron2/FastPitch dung mac dinh
    cls = ("TTS.tts.models.vits.VitsCharacters"
           if model_kind == "vits"
           else "TTS.tts.utils.text.characters.Graphemes")
    return {
        "characters_class": cls,
        "pad": "_", "eos": "~", "bos": "^", "blank": None,
        "characters": VI_CHARACTERS,
        "punctuations": VI_PUNCT,
        "phonemes": None,
    }


def dataset_block(ds_root):
    return {
        "formatter": "ljspeech",
        "dataset_name": "infore",
        "path": f"{ds_root}/infore_coqui",
        "meta_file_train": "metadata.csv",  # 3 cot: id|raw|norm (infore)
        "language": "vi",
    }


def common(name, tune, ds_root, out_sub):
    return {
        "run_name": f"{name}_infore",
        "batch_size": tune["batch"], "eval_batch_size": tune["eval_batch"],
        "mixed_precision": tune["fp16"],
        "num_loader_workers": 2, "num_eval_loader_workers": 0,
        "run_eval": False, "test_delay_epochs": -1,
        "eval_split_size": 0.05,   # 5% = ~742 cau eval, du lon
        "epochs": 1000, "print_step": 25, "save_step": 5000,
        "save_n_checkpoints": 3, "save_best_after": 5000,
        "text_cleaner": "basic_cleaners",
        "use_phonemes": False,
        # LOC do dai cho 16kHz: 1s = 16000 mau.
        # InFoRe: min 2.62s, max 11.81s -> giu 0.5s..10s de loc outlier
        "min_text_len": 5, "max_text_len": 200,
        "min_audio_len": 8000,    # 0.5s  @ 16kHz
        "max_audio_len": 160000,  # 10s   @ 16kHz
        "characters": characters_config("vits" if name == "vits" else "graph"),
        "datasets": [dataset_block(ds_root)],
        "output_path": f"models/{out_sub}/runs",
        "test_sentences": [
            "tất cả mang đến cho người xem cảm giác về sự không ranh giới",
            "bạn có thể tự bổ sung những thiếu sót của mình bằng cách tham gia các khóa học",
            "nhưng vẻ đẹp bóng đá cũng rất mong manh khi người ta sẵn sàng từ bỏ",
        ],
        "audio": {
            # 16kHz: Nyquist = 8000 Hz -> mel_fmax = 8000
            # hop_length=256 la bat buoc voi VITS: product(upsample_rates)=[8,8,2,2]=256
            # win_length=1024 @ 16kHz ~ 64ms (chap nhan duoc cho speech)
            "sample_rate": 16000,
            "fft_size": 1024, "win_length": 1024, "hop_length": 256,
            "num_mels": 80, "mel_fmin": 0.0, "mel_fmax": 8000.0,
            "signal_norm": True, "symmetric_norm": False,
            "max_norm": 1.0, "clip_norm": True,
            "do_trim_silence": True, "trim_db": 23,
        },
    }


def build_tacotron2(tune, ds_root):
    c = common("tacotron2", tune, ds_root, "tacotron2")
    c.update({"model": "tacotron2", "lr": 1e-3, "r": 1,
              "grad_clip": 5.0, "double_decoder_consistency": False,
              # tat SSIM loss -> tranh NaN; bat forward attention -> alignment
              # hoi tu nhanh hon (da xac nhan khi train VIVOS).
              "ssim_loss_alpha": 0.0,
              "use_forward_attn": True, "forward_attn_mask": True})
    return c


def build_fastpitch(tune, ds_root):
    c = common("FastSpeech2_vi", tune, ds_root, "FastSpeech2_vi")
    c.update({"model": "fast_pitch", "lr": 1e-4})
    return c


def build_vits(tune, ds_root):
    c = common("vits", tune, ds_root, "vits")
    c.update({"model": "vits", "lr_gen": 2e-4, "lr_disc": 2e-4,
              "use_speaker_embedding": False})
    return c


BUILDERS = {
    "tacotron2": build_tacotron2,
    "FastSpeech2_vi": build_fastpitch,
    "vits": build_vits,
}


def emit(name, ds_root):
    tune = GPU6[name]
    cfg = BUILDERS[name](tune, ds_root)
    out = Path("models") / name / "coqui_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {name}: {out}  (batch={tune['batch']} fp16={tune['fp16']})")
    print(f"     python -m TTS.bin.train_tts --config_path {out}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-root", default="dataset")
    ap.add_argument("--model", choices=list(BUILDERS.keys()))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not args.all and not args.model:
        ap.error("Chon --all hoac --model <ten>")
    names = list(BUILDERS.keys()) if args.all else [args.model]
    print(f"Dataset: {args.ds_root}/infore_coqui  |  16kHz  |  RTX 4050 6GB\n")
    for n in names:
        emit(n, args.ds_root)
    if args.all:
        print("=" * 56)
        print("THU TU TRAIN: 1) vits  2) tacotron2  3) FastSpeech2_vi")
        print("Train tung model MOT. So tham so THAT in ra luc khoi tao.")


if __name__ == "__main__":
    main()