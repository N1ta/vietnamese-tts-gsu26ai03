#!/usr/bin/env python3
"""
vi_normalize.py
===============
Module CHUAN HOA VAN BAN tieng Viet dung chung cho ca du an
(demo_vits, infer_long_text, eval_mos, test...).

Gom:
  - Doc so       (123 -> "một trăm hai mươi ba")
  - Viet tat     (UBND -> "ủy ban nhân dân"), doc tu dataset/data_viet_tat.json
  - Acronym NN   (AI -> "ây ai") nhom doc_chu_cai
  - Ngay/gio     (30/4/2025 -> "ngày ba mươi tháng bốn...")
  - Phan so/ngay mo ho (d/m) -> luat ngu canh (disambiguate_slash)
  - Cong thuc toan (qua math_normalize)
  - Tach cau de ngat nghi (split_for_pause), bo dau cau (strip_punct)

Ham chinh:
    normalize(text)          -> chuoi da chuan hoa (GIU dau cau ngat nghi)
    split_for_pause(text)    -> [(doan_chu, loai_nghi)]
    strip_punct(text)        -> bo het dau cau

Test:
    python vi_normalize.py --text "UBND TP.HCM họp ngày 30/4/2025, AI và 1/2."
"""

import argparse
import json
import re
import unicodedata

from math_normalize import normalize_math

# --------------------------------------------------------------------------- #
# 1) DOC SO
# --------------------------------------------------------------------------- #
_ONES = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _read_three(n, full):
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


def read_number(s):
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
        parts.append(_read_three(g, i < len(groups) - 1))
        if scales[i]:
            parts.append(scales[i])
    return " ".join(parts).strip()


def read_year(s):
    """Doc NAM DAY DU theo yeu cau user (2025 -> 'hai nghìn không trăm hai mươi
    lăm'). Truoc rut gon 'không trăm' vi ref cu hay rot duoi, nay ref 0022 sach."""
    return read_number(s)


def read_number_full(s):
    """Doc so co dau phan cach. VN: '.' = phan cach nghin (bo),
    ',' = DAU THAP PHAN -> doc 'phẩy' + tung chu so phan le.
    Vi du: 93,3 -> 'chín mươi ba phẩy ba'; 0,03 -> 'không phẩy không ba'."""
    s = s.strip()
    intpart, _, dec = s.partition(",")     # phan tach o dau phay thap phan dau tien
    intpart = intpart.replace(".", "")     # bo phan cach nghin
    out = read_number(intpart or "0")
    dec_digits = [c for c in dec if c.isdigit()]
    if dec_digits:
        out += " phẩy " + " ".join(_ONES[int(c)] for c in dec_digits)
    return out


# --------------------------------------------------------------------------- #
# 2) VIET TAT + ACRONYM NUOC NGOAI
# --------------------------------------------------------------------------- #
ENG_LETTER = {
    "a": "ây", "b": "bi", "c": "xi", "d": "đi", "e": "i", "f": "ép", "g": "gi",
    "h": "ếch", "i": "ai", "j": "giây", "k": "kây", "l": "eo", "m": "em",
    "n": "en", "o": "âu", "p": "pi", "q": "kiu", "r": "a", "s": "ét", "t": "ti",
    "u": "diu", "v": "vi", "w": "đáp liu", "x": "ích", "y": "quai", "z": "dét",
}
DOC_CHU_CAI = {
    "AI", "ML", "DL", "IT", "IOT", "CEO", "CTO", "CFO", "CPU", "GPU", "RAM",
    "USB", "GPS", "API", "SDK", "SQL", "HTML", "CSS", "PDF", "ATM", "SIM",
    "ID", "IP", "UI", "UX", "HR", "PR", "OS", "PC", "TV", "VPN", "URL", "HTTP",
}

# Viet tat BO QUA (mo rong sai trong ngu canh ky thuat/ten model):
#   GAN la thanh phan ten "HiFi-GAN" -> doc "gan", KHONG doc "mạng đối sinh".
SKIP_ABBR = {"GAN"}

ABBR_JSON = "dataset/data_viet_tat.json"
ABBR_FALLBACK = {
    "UBND": "ủy ban nhân dân", "TP.HCM": "thành phố hồ chí minh",
    "CNTT": "công nghệ thông tin", "THPT": "trung học phổ thông",
    "GS": "giáo sư", "TS": "tiến sĩ",
}


def read_as_eng_letters(abbr):
    """AI -> 'ây ai'."""
    return " ".join(ENG_LETTER[c.lower()] for c in abbr if c.lower() in ENG_LETTER)


def load_abbr(path=ABBR_JSON, include_slang=False):
    """Doc JSON viet tat -> dict {tu: cach_doc}. Loc slang, xu ly doc_chu_cai."""
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return dict(sorted(ABBR_FALLBACK.items(), key=lambda kv: -len(kv[0])))
    table = {}
    for x in data:
        k = str(x.get("tu_viet_tat", "")).strip()
        loai = x.get("loai", "")
        if not k or k in SKIP_ABBR:
            continue
        if loai == "chat_slang" and not include_slang:
            continue
        if loai == "doc_chu_cai" or k in DOC_CHU_CAI:
            table[k] = read_as_eng_letters(k)
        else:
            v = str(x.get("day_du", "")).strip()
            if v:
                table[k] = v
    return dict(sorted(table.items(), key=lambda kv: -len(kv[0])))


ABBR = load_abbr()

# --------------------------------------------------------------------------- #
# 3) NGAY / PHAN SO MO HO
# --------------------------------------------------------------------------- #
DATE_KEYWORDS = ["ngày", "mùng", "mồng", "hôm", "vào", "dịp", "lễ", "hạn", "sinh nhật"]


def disambiguate_slash(text):
    """d/m (hai so nho) -> NGAY hoac PHAN SO theo luat ngu canh 3 tang."""
    def repl(m):
        a, b = int(m.group(1)), int(m.group(2))
        if a == 0 or b == 0 or b > 12 or a > 31:          # tang 1: chac chan phan so
            return f"{m.group(1)} phần {m.group(2)}"
        before = text[max(0, m.start() - 20):m.start()].lower()
        if any(kw in before for kw in DATE_KEYWORDS):       # tang 2: ngu canh ngay
            return f" ngày {m.group(1)} tháng {m.group(2)} "
        return f"{m.group(1)} phần {m.group(2)}"            # tang 3: mac dinh phan so
    return re.sub(r"\b(\d{1,2})\s*/\s*(\d{1,2})\b", repl, text)


# --------------------------------------------------------------------------- #
# 4) DAU CAU NGAT NGHI
# --------------------------------------------------------------------------- #
PAUSE_LONG = ".!?…"
PAUSE_SHORT = ",;:"

SYMBOL_MAP = {  # ky hieu khong thuoc toan
    "&": " và ", "$": " đô la ", "~": " xấp xỉ ", "°": " độ ",
}

# Tu ngoai lai F5 (don ngu tieng Viet) doc re/vo -> phien am tieng Viet cho ro.
FOREIGN_READ = {
    "robot": "rô bốt",
}

# Don vi dung luong -> doc phien am (HOA, phan biet chu hoa thuong).
UNIT_READ = {
    "GB": "gi ga bai", "MB": "mê ga bai", "KB": "ki lô bai", "TB": "tê ra bai",
}


def normalize(text, abbr=None):
    """Chuan hoa day du, GIU dau cau (. , ! ? ; :) de lam diem ngat nghi."""
    if abbr is None:
        abbr = ABBR
    t = unicodedata.normalize("NFC", text)

    # tu ngoai lai -> phien am tieng Viet (truoc moi buoc khac)
    for w, r in FOREIGN_READ.items():
        t = re.sub(rf"(?i)\b{re.escape(w)}\b", r, t)
    # don vi dung luong (GB->gi ga bai...) - phan biet HOA, truoc doc so
    for w, r in UNIT_READ.items():
        t = re.sub(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", f" {r} ", t)

    # ngay co nam (bat luon "ngày" phia truoc neu co, tranh lap)
    t = re.sub(r"(?:ngày\s+)?\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b",
               lambda m: f" ngày {m.group(1)} tháng {m.group(2)} năm {read_year(m.group(3))} ",
               t, flags=re.IGNORECASE)
    # gio:phut
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b",
               lambda m: f" {m.group(1)} giờ {m.group(2)} phút ", t)
    # d/m mo ho -> ngay hoac phan so theo ngu canh
    t = disambiguate_slash(t)
    # cong thuc toan (phan so con lai, dau toan, bien so x->ích) -- TRUOC viet tat
    t = normalize_math(t)
    # viet tat (match HOA) -> dang day du / doc chu cai
    for k, v in abbr.items():
        if k not in t:          # bo qua nhanh: chi chay regex cho viet tat THUC SU co trong text
            continue            # (tang toc lon khi tu dien lon ~2400 muc)
        if "." in k:
            t = t.replace(k, f" {v} ")
        else:
            t = re.sub(rf"(?<![A-Za-zÀ-ỹ]){re.escape(k)}(?![A-Za-zÀ-ỹ])", f" {v} ", t)
    # ky hieu khac
    for sym, read in SYMBOL_MAP.items():
        t = t.replace(sym, read)
    # so -> chu (',' = thap phan -> 'phẩy'; '.' = phan cach nghin -> bo)
    t = re.sub(r"\d+(?:[.,]\d+)*",
               lambda m: f" {read_number_full(m.group(0))} ", t)
    # bo ky tu la, GIU chu + khoang trang + dau cau ngat nghi
    keep = re.escape(PAUSE_LONG + PAUSE_SHORT)
    t = re.sub(rf"[^A-Za-zÀ-ỹ\s{keep}]", " ", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(rf"\s+([{keep}])", r"\1", t)   # bo khoang trang THUA truoc dau cau
    t = t.strip().lower()
    t = t.replace("tháng bốn", "tháng tư")     # thang 4 doc "tư" (tu nhien hon "bốn")
    return t


def strip_punct(seg):
    """Bo het dau cau (model vocab chi co chu + space)."""
    s = re.sub(rf"[{re.escape(PAUSE_LONG + PAUSE_SHORT)}]", " ", seg)
    return re.sub(r"\s+", " ", s).strip().lower()


def split_for_pause(text):
    """Tach text (da normalize) -> [(doan_chu, loai_nghi)] theo dau cau."""
    parts = re.split(rf"([{re.escape(PAUSE_LONG + PAUSE_SHORT)}])", text)
    segs, buf = [], ""
    for p in parts:
        if p in PAUSE_LONG:
            if buf.strip():
                segs.append((strip_punct(buf), "long"))
            buf = ""
        elif p in PAUSE_SHORT:
            if buf.strip():
                segs.append((strip_punct(buf), "short"))
            buf = ""
        else:
            buf += p
    if buf.strip():
        segs.append((strip_punct(buf), "none"))
    return [(s, k) for s, k in segs if s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--slang", action="store_true", help="Bat nhom chat_slang")
    args = ap.parse_args()
    abbr = load_abbr(include_slang=True) if args.slang else ABBR
    print("RAW :", args.text)
    print("NORM:", normalize(args.text, abbr))
    print("SEGS:", split_for_pause(normalize(args.text, abbr)))


if __name__ == "__main__":
    main()
