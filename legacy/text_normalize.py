#!/usr/bin/env python3
"""
text_normalize.py
=================
Chuan hoa van ban tieng Viet truoc khi G2P / train TTS:
  - Thay tu viet tat  (UBND -> Uy ban Nhan dan)   [dung data nhom + bo sung]
  - Doc so            (123  -> mot tram hai muoi ba)
  - Doc ky hieu       (%    -> phan tram,  +  -> cong, ...)
  - Doc ngay/gio don gian, don vi do luong co ban
  - Bo ky tu rac, gom khoang trang, ve chu thuong

Day la buoc CHUNG cho ca hai che do char va phoneme. Luon chay truoc g2p_vi.py.

Cach dung:
    # chuan hoa 1 filelist (LJSpeech "wav|text") -> ghi cot text da normalize
    python text_normalize.py \
        --in  dataset/single_speaker/VIVOSSPK35/metadata_ljspeech.csv \
        --out dataset/single_speaker/VIVOSSPK35/metadata_norm.csv \
        --abbr data_viet_tat.json

    # hoac chuan hoa 1 chuoi de test:
    python text_normalize.py --text "UBND TPHCM chi 25% cho BV vao 1/5/2026"
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1) Tu viet tat: hat nhan tu data nhom + bo sung pho bien.
#    File ngoai (--abbr) se merge de len, de ban mo rong sau nay.
# --------------------------------------------------------------------------- #
ABBR_BUILTIN = {
    # hanh chinh / to chuc
    "UBND": "Ủy ban Nhân dân", "HĐND": "Hội đồng Nhân dân",
    "BYT": "Bộ Y tế", "BGDĐT": "Bộ Giáo dục và Đào tạo",
    "TPHCM": "Thành phố Hồ Chí Minh", "TP.HCM": "Thành phố Hồ Chí Minh",
    "TP": "Thành phố", "TT": "Thị trấn",
    # ky thuat
    "CNTT": "Công nghệ Thông tin", "CSDL": "Cơ sở dữ liệu",
    "AI": "Trí tuệ Nhân tạo", "ML": "Học máy",
    # y te
    "BV": "Bệnh viện", "BS": "Bác sĩ",
    # giao duc
    "SV": "sinh viên", "GV": "giáo viên",
    "THPT": "Trung học Phổ thông", "THCS": "Trung học Cơ sở",
    "ĐH": "Đại học", "CĐ": "Cao đẳng",
    # giao thong
    "QL": "Quốc lộ", "TL": "Tỉnh lộ",
    # tai chinh
    "VNĐ": "Việt Nam đồng", "VND": "Việt Nam đồng",
    "USD": "đô la Mỹ", "BHXH": "Bảo hiểm Xã hội",
    # hoc ham / hoc vi
    "GS": "Giáo sư", "PGS": "Phó Giáo sư", "TS": "Tiến sĩ", "ThS": "Thạc sĩ",
    # chat / mang xa hoi
    "ko": "không", "k": "không", "dc": "được", "đc": "được",
    "nt": "nhắn tin", "mk": "mình", "bt": "bình thường", "vs": "với",
}

# --------------------------------------------------------------------------- #
# 2) Ky hieu -> chu doc
# --------------------------------------------------------------------------- #
SYMBOL_MAP = {
    "%": " phần trăm ", "+": " cộng ", "=": " bằng ",
    "&": " và ", "@": " a còng ", "#": " thăng ",
    "°": " độ ", "²": " bình phương ", "³": " lập phương ",
    "$": " đô la ", "€": " ơ rô ", "£": " bảng ",
    "×": " nhân ", "÷": " chia ", "<": " nhỏ hơn ", ">": " lớn hơn ",
    "~": " xấp xỉ ", "±": " cộng trừ ",
}

UNIT_MAP = {
    "km": "ki lô mét", "kg": "ki lô gam", "kwh": "ki lô oát giờ",
    "cm": "xăng ti mét", "mm": "mi li mét", "m2": "mét vuông",
    "ml": "mi li lít", "kb": "ki lô bai", "mb": "mê ga bai",
    "gb": "gi ga bai", "tb": "tê ra bai",
}

# --------------------------------------------------------------------------- #
# 3) Doc so tieng Viet
# --------------------------------------------------------------------------- #
_ONES = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _read_three(n: int, full: bool) -> str:
    """Doc khoi 3 chu so (0..999). full=True khi co khoi cao hon dung truoc."""
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
    """Doc so nguyen (chuoi chu so) thanh tieng Viet."""
    s = s.lstrip("0") or "0"
    if s == "0":
        return "không"
    n = int(s)
    if n == 0:
        return "không"
    groups, scales = [], ["", "nghìn", "triệu", "tỷ"]
    while n > 0:
        n, r = divmod(n, 1000)
        groups.append(r)
    parts = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        full = i < len(groups) - 1  # khong phai khoi cao nhat -> doc du 3 chu so
        parts.append(_read_three(g, full))
        if scales[i]:
            parts.append(scales[i])
    return " ".join(parts).strip()


def _num_repl(m: re.Match) -> str:
    raw = m.group(0)
    # tieng Viet: '.' phan tach hang nghin, ',' phan tach thap phan
    if "," in raw:
        a, _, b = raw.partition(",")
        a = a.replace(".", "")
        digits = " ".join(_ONES[int(c)] for c in b if c.isdigit())
        return f"{read_number(a)} phẩy {digits}"
    # khong co dau phay -> moi dau cham la phan tach nghin
    return read_number(raw.replace(".", ""))


# --------------------------------------------------------------------------- #
def load_abbr(extra_path):
    table = dict(ABBR_BUILTIN)
    if extra_path and Path(extra_path).exists():
        data = json.loads(Path(extra_path).read_text(encoding="utf-8"))
        if isinstance(data, list):  # dinh dang nhom: list of records
            for x in data:
                k = str(x.get("tu_viet_tat", "")).strip()
                v = str(x.get("day_du", "")).strip()
                if k and v:
                    table[k] = v
        elif isinstance(data, dict):
            table.update({str(k): str(v) for k, v in data.items()})
    # sap xep key dai truoc de thay the uu tien cum dai
    return dict(sorted(table.items(), key=lambda kv: -len(kv[0])))


def normalize(text: str, abbr: dict) -> str:
    t = unicodedata.normalize("NFC", text)

    # ngay dang d/m/yyyy hoac d-m-yyyy
    def _date(m):
        d, mo, y = m.group(1), m.group(2), m.group(3)
        return f" ngày {read_number(d)} tháng {read_number(mo)} năm {read_number(y)} "
    t = re.sub(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", _date, t)

    # gio dang h:m
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b",
               lambda m: f" {read_number(m.group(1))} giờ {read_number(m.group(2))} phút ", t)

    # viet tat: match nguyen tu (co bien gioi). Key co dau cham xu ly rieng.
    for k, v in abbr.items():
        if any(ch in k for ch in ".+"):
            t = t.replace(k, f" {v} ")
        else:
            t = re.sub(rf"(?<![\wÀ-ỹ]){re.escape(k)}(?![\wÀ-ỹ])", f" {v} ", t)

    # don vi do luong (sau so): "5km" / "5 km"
    for u, read in UNIT_MAP.items():
        t = re.sub(rf"(\d)\s*{u}\b", rf"\1 {read} ", t, flags=re.IGNORECASE)

    # ky hieu
    for sym, read in SYMBOL_MAP.items():
        t = t.replace(sym, read)

    # so (sau khi da xu ly ngay/gio/don vi) — them space de tranh dính vào chữ
    t = re.sub(r"\d+(?:[.,]\d+)*", lambda m: f" {_num_repl(m)} ", t)

    # bo ky tu khong phai chu/khoang trang tieng Viet
    t = re.sub(r"[^0-9A-Za-zÀ-ỹ\s]", " ", t)
    # gom khoang trang, ve chu thuong
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


# --------------------------------------------------------------------------- #
def process_filelist(in_path, out_path, abbr):
    lines_out = []
    n = 0
    for line in Path(in_path).read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("|")
        if parts[0].endswith(".wav"):           # LJSpeech: wav|text
            wav, text = parts[0], "|".join(parts[1:])
            lines_out.append(f"{wav}|{normalize(text, abbr)}")
        else:                                     # FS2: name|spk|{ph}|raw
            raw = parts[-1]
            norm = normalize(raw, abbr)
            parts[-1] = norm
            parts[2] = "{" + norm + "}"
            lines_out.append("|".join(parts))
        n += 1
    Path(out_path).write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"[done] normalized {n} lines -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile")
    ap.add_argument("--out", dest="outfile")
    ap.add_argument("--abbr", default=None, help="JSON viet tat nhom (optional).")
    ap.add_argument("--text", default=None, help="Chuan hoa 1 chuoi de test.")
    args = ap.parse_args()

    abbr = load_abbr(args.abbr)
    print(f"[init] {len(abbr)} abbreviations loaded")

    if args.text is not None:
        print("RAW :", args.text)
        print("NORM:", normalize(args.text, abbr))
        return
    if not args.infile or not args.outfile:
        ap.error("Can --in va --out (hoac dung --text).")
    process_filelist(args.infile, args.outfile, abbr)


if __name__ == "__main__":
    main()
