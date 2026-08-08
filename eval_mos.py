#!/usr/bin/env python3
"""
eval_mos.py
===========
Sinh tap audio test chung cho ca 3 model de danh gia MOS.

Buoc 1 — sinh audio (sau khi co checkpoint):
    python eval_mos.py --gen \
        --vits-ckpt     models/vits/runs/vits_infore-.../best_model.pth \
        --taco-ckpt     models/tacotron2/runs/tacotron2_infore-.../best_model.pth \
        --fp-ckpt       models/FastSpeech2_vi/runs/FastSpeech2_vi_infore-.../best_model.pth \
        --out-dir       eval/mos_audio/

Buoc 2 — in bang ket qua MOS (sau khi co diem tu nguoi nghe):
    python eval_mos.py --report --scores eval/mos_scores.csv

Cau truc eval/mos_audio/:
    vits/   sent01.wav ... sent10.wav
    taco/   sent01.wav ... sent10.wav
    fp/     sent01.wav ... sent10.wav
"""

import argparse
import csv
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 10 cau test chung — doi nghia, do dai trung binh, khong co so phuc tap
# ---------------------------------------------------------------------------
TEST_SENTENCES = [
    "trí tuệ nhân tạo đang thay đổi cách con người giao tiếp với máy tính",
    "tổng hợp giọng nói tiếng việt là bài toán thú vị và có nhiều ứng dụng thực tiễn",
    "mô hình học sâu giúp cải thiện chất lượng âm thanh một cách đáng kể",
    "người dùng có thể nghe nội dung văn bản dài mà không cần đọc trực tiếp",
    "công nghệ này được ứng dụng trong trợ lý ảo và hệ thống đọc sách điện tử",
    "chúng tôi đã thu thập và xử lý hơn mười bốn nghìn câu tiếng việt",
    "kết quả thực nghiệm cho thấy mô hình đạt được chất lượng giọng nói tự nhiên",
    "dữ liệu huấn luyện bao gồm nhiều loại câu từ đơn giản đến phức tạp",
    "giọng nói tổng hợp cần nghe rõ ràng tự nhiên và dễ hiểu đối với người nghe",
    "ba kiến trúc được so sánh là tacotron hai fastspeech hai và vits",
]

MODELS = {
    "vits": "VITS",
    "taco": "Tacotron2",
    "fp":   "FastPitch (FastSpeech2)",
}


def gen_audio(args):
    """Sinh audio cho ca 3 model."""
    from TTS.api import TTS
    from TTS.config import load_config

    out_dir = Path(args.out_dir)
    checkpoints = {
        "vits": (args.vits_ckpt, "models/vits/coqui_config.json"),
        "taco": (args.taco_ckpt, "models/tacotron2/coqui_config.json"),
        "fp":   (args.fp_ckpt,   "models/FastSpeech2_vi/coqui_config.json"),
    }

    rtf_data = {}

    for model_key, (ckpt, cfg_path) in checkpoints.items():
        if not ckpt:
            print(f"[skip] {model_key}: khong co checkpoint")
            continue
        if not Path(ckpt).exists():
            print(f"[skip] {model_key}: khong tim thay {ckpt}")
            continue

        model_dir = out_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[load] {MODELS[model_key]}...")
        tts = TTS(model_path=ckpt, config_path=cfg_path)
        tts.to("cuda")
        cfg = load_config(cfg_path)
        sr = cfg.audio.sample_rate

        total_audio = 0.0
        total_wall = 0.0

        for i, sent in enumerate(TEST_SENTENCES, 1):
            out_wav = model_dir / f"sent{i:02d}.wav"
            t0 = time.time()
            tts.tts_to_file(text=sent, file_path=str(out_wav))
            elapsed = time.time() - t0

            import soundfile as sf
            wav, _ = sf.read(str(out_wav))
            dur = len(wav) / sr
            total_audio += dur
            total_wall += elapsed
            print(f"  [{i:2d}] {dur:.1f}s audio / {elapsed:.1f}s wall")

        rtf = total_wall / total_audio
        rtf_data[model_key] = rtf
        print(f"  -> RTF {MODELS[model_key]}: {rtf:.3f}")

    print(f"\n[done] Audio da luu vao: {out_dir}")
    print("\nSo sanh RTF (Real-Time Factor, nho hon la tot hon):")
    for k, v in rtf_data.items():
        print(f"  {MODELS[k]:25s}: RTF = {v:.3f}")
    print("\nTiep theo: cho nguoi nghe danh gia MOS (1-5) tren tung file audio,")
    print("roi ghi vao eval/mos_scores.csv va chay:")
    print("    python eval_mos.py --report --scores eval/mos_scores.csv")


def report(args):
    """In bang tong hop MOS tu file CSV."""
    path = Path(args.scores)
    if not path.exists():
        raise SystemExit(f"Khong tim thay: {path}")

    # Format CSV: model,sentence,naturalness,clarity
    # (model: vits/taco/fp; naturalness/clarity: 1-5)
    data = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = row["model"]
            nat = float(row["naturalness"])
            cla = float(row.get("clarity", nat))
            if m not in data:
                data[m] = {"naturalness": [], "clarity": []}
            data[m]["naturalness"].append(nat)
            data[m]["clarity"].append(cla)

    print("\n" + "="*60)
    print(f"{'Model':<25} {'MOS Tự nhiên':>14} {'MOS Rõ ràng':>12} {'Mẫu':>6}")
    print("-"*60)
    for m, vals in sorted(data.items()):
        n = vals["naturalness"]
        c = vals["clarity"]
        avg_n = sum(n) / len(n)
        avg_c = sum(c) / len(c)
        print(f"{MODELS.get(m, m):<25} {avg_n:>12.2f} {avg_c:>12.2f} {len(n):>6}")
    print("="*60)
    print("\n(Thang MOS: 1=Rat kem  2=Kem  3=Trung binh  4=Tot  5=Tuyet voi)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true", help="Sinh audio test")
    ap.add_argument("--report", action="store_true", help="In ket qua MOS")

    # --gen options
    ap.add_argument("--vits-ckpt", default=None)
    ap.add_argument("--taco-ckpt", default=None)
    ap.add_argument("--fp-ckpt",   default=None)
    ap.add_argument("--out-dir",   default="eval/mos_audio")

    # --report options
    ap.add_argument("--scores", default="eval/mos_scores.csv")

    args = ap.parse_args()

    if args.gen:
        gen_audio(args)
    elif args.report:
        report(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
