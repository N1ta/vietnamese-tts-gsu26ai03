#!/usr/bin/env python3
"""
make_finetune_init.py
=====================
Tao checkpoint khoi tao cho fine-tune Tacotron2 tieng Viet tu checkpoint
tieng Anh (LJSpeech DDC).

Van de: vocab tieng Anh (~60 ky tu) != 93 ky tu tieng Viet. Layer embedding
co kich thuoc khac nhau -> restore truc tiep se loi shape mismatch.

Giai phap (partial restore): nap checkpoint goc, BO cac layer phu thuoc vocab
(embedding + cac layer co so chieu = so ky tu), giu lai phan "biet tao giong"
(encoder conv/lstm, decoder, attention, postnet). Luu thanh .pth moi de dung
voi:  python -m TTS.bin.train_tts --config_path ... --restore_path <file nay>

Cach dung:
    python make_finetune_init.py ^
        --src "C:/Users/Nita/AppData/Local/tts/tts_models--en--ljspeech--tacotron2-DDC/model_file.pth" ^
        --out models/tacotron2/ft_init.pth

Sau do train:
    python -m TTS.bin.train_tts --config_path models/tacotron2/coqui_config.json --restore_path models/tacotron2/ft_init.pth
"""

import argparse
import torch


# Cac tu khoa layer phu thuoc vocab -> can BO khi doi ngon ngu
VOCAB_DEPENDENT = ["embedding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="model_file.pth tieng Anh")
    ap.add_argument("--out", required=True, help="checkpoint khoi tao moi")
    args = ap.parse_args()

    print(f"[load] {args.src}")
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)

    # Coqui luu trong so o key 'model'
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    else:
        state = ckpt

    kept, dropped = {}, []
    for k, v in state.items():
        if any(tok in k for tok in VOCAB_DEPENDENT):
            dropped.append(k)
            continue
        kept[k] = v

    print(f"[info] giu {len(kept)} tensor, bo {len(dropped)} tensor phu thuoc vocab:")
    for k in dropped:
        print(f"        - {k}  shape={tuple(state[k].shape)}")

    # Luu checkpoint moi: them metadata Coqui trainer can khi restore.
    # step/epoch = None de trainer hieu day la fine-tune (bat dau lai tu 0),
    # khong phai resume. optimizer/scaler = None de khoi tao moi.
    new_ckpt = {
        "model": kept,
        "step": 0,
        "epoch": 0,
        "optimizer": None,
        "scaler": None,
        "config": ckpt.get("config", None) if isinstance(ckpt, dict) else None,
    }
    torch.save(new_ckpt, args.out)
    print(f"[done] luu checkpoint khoi tao -> {args.out}")
    print("Luu y: khi train, Coqui se bao 'missing keys' cho embedding -> BINH THUONG,")
    print("nghia la layer do hoc lai tu dau cho ky tu tieng Viet.")


if __name__ == "__main__":
    main()