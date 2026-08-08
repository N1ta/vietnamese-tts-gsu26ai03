#!/usr/bin/env python3
"""
prepare_infore.py
=================
Chuan bi dataset InFoRe 16kHz cho Coqui-TTS (LJSpeech formatter).

Lam 3 viec:
  1. Doc toan bo <id>.txt tu infore_16k_denoised_2/
  2. Ap text normalization (so->chu, ky hieu, viet tat)
  3. Ghi metadata.csv (dinh dang 3 cot: id|raw|normalized)
  4. Tao wavs/ = NTFS junction tro toi thu muc goc
     (tranh copy ~2.8 GB, KHONG can quyen admin tren Windows 10+)

Cau truc dich:
    dataset/infore_coqui/
        metadata.csv
        wavs/  <-- junction -> dataset/infore_16k_denoised_2/

Cach dung:
    python prepare_infore.py
    python prepare_infore.py --src dataset/infore_16k_denoised_2 --out dataset/infore_coqui
"""

import argparse
import os
import re
import subprocess
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Text normalization (inline, khong phu thuoc file ngoai)
# ---------------------------------------------------------------------------

_ONES = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _read_three(n: int, full: bool) -> str:
    tram, du = divmod(n, 100)
    chuc, dv = divmod(du, 10)
    out = []
    if tram > 0 or full:
        out += [_ONES[tram], "trăm"]
    if chuc == 0:
        if dv > 0 and (tram > 0 or full):
            out.append("lẻ")
        if dv > 0:
            out.append(_ONES[dv])
    elif chuc == 1:
        out.append("mười")
        if dv == 5:
            out.append("lăm")
        elif dv > 0:
            out.append(_ONES[dv])
    else:
        out += [_ONES[chuc], "mươi"]
        if dv == 1:
            out.append("mốt")
        elif dv == 5:
            out.append("lăm")
        elif dv > 0:
            out.append(_ONES[dv])
    return " ".join(out)


def read_number(s: str) -> str:
    s = s.lstrip("0") or "0"
    if s == "0":
        return "không"
    n = int(s)
    groups, scales = [], ["", "nghìn", "triệu", "tỷ"]
    while n > 0:
        n, r = divmod(n, 1000)
        groups.append(r)
    parts = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        full = i < len(groups) - 1
        parts.append(_read_three(g, full))
        if scales[i]:
            parts.append(scales[i])
    return " ".join(parts).strip()


def _num_repl(m: re.Match) -> str:
    raw = m.group(0)
    if "," in raw:
        a, _, b = raw.partition(",")
        a = a.replace(".", "")
        digits = " ".join(_ONES[int(c)] for c in b if c.isdigit())
        return f"{read_number(a)} phẩy {digits}"
    return read_number(raw.replace(".", ""))


ABBR = {
    "UBND": "Ủy ban Nhân dân", "HĐND": "Hội đồng Nhân dân",
    "BYT": "Bộ Y tế", "BGDĐT": "Bộ Giáo dục và Đào tạo",
    "TPHCM": "Thành phố Hồ Chí Minh", "TP.HCM": "Thành phố Hồ Chí Minh",
    "TP": "Thành phố", "TT": "Thị trấn",
    "CNTT": "Công nghệ Thông tin", "AI": "Trí tuệ Nhân tạo",
    "BV": "Bệnh viện", "BS": "Bác sĩ",
    "SV": "sinh viên", "GV": "giáo viên",
    "THPT": "Trung học Phổ thông", "THCS": "Trung học Cơ sở",
    "ĐH": "Đại học", "CĐ": "Cao đẳng",
    "VNĐ": "Việt Nam đồng", "VND": "Việt Nam đồng",
    "USD": "đô la Mỹ", "GS": "Giáo sư", "PGS": "Phó Giáo sư",
    "TS": "Tiến sĩ", "ThS": "Thạc sĩ",
}
ABBR = dict(sorted(ABBR.items(), key=lambda kv: -len(kv[0])))

SYMBOL_MAP = {
    "%": " phần trăm ", "+": " cộng ", "=": " bằng ", "&": " và ",
    "@": " a còng ", "°": " độ ", "²": " bình phương ", "³": " lập phương ",
    "$": " đô la ", "×": " nhân ", "÷": " chia ", "~": " xấp xỉ ", "±": " cộng trừ ",
}

UNIT_MAP = {
    "km": "ki lô mét", "kg": "ki lô gam", "cm": "xăng ti mét",
    "mm": "mi li mét", "ml": "mi li lít", "m2": "mét vuông",
    "kwh": "ki lô oát giờ", "gb": "gi ga bai", "mb": "mê ga bai",
}


def normalize(text: str, keep_punct: bool = False) -> str:
    t = unicodedata.normalize("NFC", text)

    # Ngay thang: d/m/yyyy
    def _date(m):
        d, mo, y = m.group(1), m.group(2), m.group(3)
        return f" ngày {read_number(d)} tháng {read_number(mo)} năm {read_number(y)} "
    t = re.sub(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", _date, t)

    # Gio phut
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b",
               lambda m: f" {read_number(m.group(1))} giờ {read_number(m.group(2))} phút ", t)

    # Viet tat
    for k, v in ABBR.items():
        if "." in k:
            t = t.replace(k, f" {v} ")
        else:
            t = re.sub(rf"(?<![a-zA-ZÀ-ỹ]){re.escape(k)}(?![a-zA-ZÀ-ỹ])", f" {v} ", t)

    # Don vi do luong
    for u, read in UNIT_MAP.items():
        t = re.sub(rf"(\d)\s*{u}\b", rf"\1 {read} ", t, flags=re.IGNORECASE)

    # Ky hieu
    for sym, read in SYMBOL_MAP.items():
        t = t.replace(sym, read)

    # So (sau xu ly ngay/gio/don vi)
    t = re.sub(r"\d+(?:[.,]\d+)*", _num_repl, t)

    if keep_punct:
        # GIU dau cau ngat nghi (. , ! ? ; :) de model hoc nhip dieu
        keep = re.escape(".,!?;:")
        t = re.sub(rf"[^A-Za-zÀ-ỹ\s{keep}]", " ", t)
        t = re.sub(r"\s+", " ", t).strip().lower()
        # bo khoang trang truoc dau cau ("dep ," -> "dep,")
        t = re.sub(r"\s+([.,!?;:])", r"\1", t)
        # gop dau cau lap ("..." -> ".")
        t = re.sub(r"([.,!?;:])\1+", r"\1", t)
    else:
        # Bo het dau cau, chi giu chu + khoang trang (mac dinh, an toan)
        t = re.sub(r"[^A-Za-zÀ-ỹ\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip().lower()
    return t


# ---------------------------------------------------------------------------

def create_junction(link: Path, target: Path):
    """Tao NTFS junction tren Windows (khong can admin)."""
    if link.exists() or link.is_symlink():
        print(f"  [skip] {link} da ton tai")
        return
    # mklink /J link target
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  [junction] {link} -> {target}")
    else:
        print(f"  [warn] khong tao duoc junction: {result.stderr.strip()}")
        print(f"         Thu tao thu muc wavs/ va copy thu cong neu can.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"dataset/infore_16k_denoised_2",
                    help="Thu muc nguon InFoRe")
    ap.add_argument("--out", default=r"dataset/infore_coqui",
                    help="Thu muc dich")
    ap.add_argument("--no-norm", action="store_true",
                    help="Khong chay text normalization (chi lowercase)")
    ap.add_argument("--keep-punct", action="store_true",
                    help="GIU dau cau (. , ! ?) -> ghi metadata_punct.csv (de train lai cho model hoc ngat nghi)")
    ap.add_argument("--check", action="store_true",
                    help="Chi kiem tra, khong ghi file")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    if not src.exists():
        raise SystemExit(f"[error] Khong tim thay: {src}")

    # Thu thap tat ca id
    wav_ids = {f.stem for f in src.iterdir() if f.suffix == ".wav"}
    txt_ids = {f.stem for f in src.iterdir() if f.suffix == ".txt"}
    valid_ids = sorted(wav_ids & txt_ids)
    print(f"[info] Tim thay {len(valid_ids)} cap wav/txt hop le trong {src}")

    if args.check:
        print("[check] Khong ghi file (--check mode).")
        return

    # Tao thu muc dich
    out.mkdir(parents=True, exist_ok=True)

    # Tao junction wavs/
    wavs_link = out / "wavs"
    create_junction(wavs_link, src.resolve())

    # Tao metadata (ten file khac nhau theo che do)
    meta_name = "metadata_punct.csv" if args.keep_punct else "metadata.csv"
    meta_path = out / meta_name
    lines = []
    skipped = 0
    norm_changed = 0

    for fid in valid_ids:
        txt_path = src / (fid + ".txt")
        try:
            raw = txt_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"  [skip] {fid}: {e}")
            skipped += 1
            continue

        if not raw:
            skipped += 1
            continue

        if args.no_norm:
            norm = raw.lower()
        else:
            norm = normalize(raw, keep_punct=args.keep_punct)

        if norm != raw.lower():
            norm_changed += 1

        # 3 cot: id|raw|normalized  (raw giu nguyen, normalized da chuan hoa)
        lines.append(f"{fid}|{raw}|{norm}")

    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[done] Ghi {len(lines)} dong -> {meta_path}")
    print(f"       Bo qua: {skipped}")
    print(f"       Text bi doi sau normalize: {norm_changed} ({norm_changed*100//max(len(lines),1)}%)")
    print(f"       Junction wavs/: {wavs_link}")
    print(f"\nKiem tra 5 mau:")
    for line in lines[:5]:
        fid, raw, norm = line.split("|", 2)
        print(f"  [{fid}]")
        print(f"    raw : {raw[:80]}")
        if norm != raw:
            print(f"    norm: {norm[:80]}")
    print(f"\nCau hinh Coqui can dung:")
    print(f"  path: {out.resolve()}")
    print(f"  meta_file_train: metadata.csv")
    print(f"  wavs/  -> {src.resolve()}")


if __name__ == "__main__":
    main()
