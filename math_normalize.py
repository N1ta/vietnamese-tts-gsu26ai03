#!/usr/bin/env python3
"""
math_normalize.py
=================
Chuan hoa CONG THUC TOAN co ban -> chu doc tieng Viet (yeu cau giam thi #2).

Lam (phan co ban, chua xu ly ham P(A)/f(x)):
  - Phan so:    a/b        -> "a phần b"
  - Luy thua:   x^2, x²    -> "x mũ hai"
  - Can:        √x         -> "căn x"
  - Dau toan:   + - × ÷ =  -> cộng / trừ / nhân / chia / bằng
  - So sanh:    < > ≤ ≥ ≠  -> nhỏ hơn / lớn hơn / ... / khác
  - Bien so:    x y z a b  -> tên chữ cái tiếng Việt (ích, i, dét, a, bê...)
  - Ngoac:      { } [ ]    -> bỏ (giữ nội dung)

KHONG doc so o day (de text_normalize/prepare lo). Chi bien doi ky hieu/bien.
Goi normalize_math() TRUOC buoc doc so.

Test nhanh:
    python math_normalize.py --text "Tìm x, biết 56 - x = 1236/5000 và x² > 4"
"""

import argparse
import re

# Ten chu cai tieng Viet (kieu doc trong lop hoc toan)
LETTER_NAMES = {
    "a": "a",   "b": "bê",  "c": "xê",  "d": "đê",  "e": "e",
    "f": "ép",  "g": "gờ",  "h": "hát", "i": "i",   "j": "giây",
    "k": "ca",  "l": "lờ",  "m": "mờ",  "n": "nờ",  "o": "o",
    "p": "pê",  "q": "quy", "r": "rờ",  "s": "ét",  "t": "tê",
    "u": "u",   "v": "vê",  "w": "vê kép", "x": "ích", "y": "i",
    "z": "dét",
}

# Ky hieu toan -> chu (dau '-' xu ly rieng ben duoi vi de nham gach noi)
MATH_SYMBOLS = {
    "+": " cộng ", "×": " nhân ", "*": " nhân ", "·": " nhân ",
    "÷": " chia ", "=": " bằng ", "≠": " khác ",
    "≤": " nhỏ hơn hoặc bằng ", "≥": " lớn hơn hoặc bằng ",
    "<": " nhỏ hơn ", ">": " lớn hơn ",
    "±": " cộng trừ ", "√": " căn ", "%": " phần trăm ",
    "°": " độ ",
}

# Chi so tren (superscript) -> mu
SUPERSCRIPT = {
    "⁰": " mũ không ", "¹": " mũ một ", "²": " mũ hai ", "³": " mũ ba ",
    "⁴": " mũ bốn ", "⁵": " mũ năm ", "⁶": " mũ sáu ", "⁷": " mũ bảy ",
    "⁸": " mũ tám ", "⁹": " mũ chín ",
}

_POW_WORDS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


# Dau hieu NGU CANH TOAN: chi doi chu cai don -> bien so khi cau co toan,
# tranh "Bộ Y tế"->"bộ i tế", "vitamin C"->"vitamin xê" trong van xuoi.
_MATH_CONTEXT = re.compile(
    r"[=+×÷*<>^²³√≤≥≠±]"          # ky hieu toan
    r"|\d\s*/\s*\d"                # phan so
    r"|\d\s*-\s*[A-Za-z(]"         # so trừ bien/ngoac
    r"|[A-Za-z)]\s*-\s*\d"         # bien trừ so
)


def _normalize_clause(t: str) -> str:
    """Chuan hoa toan cho 1 CAU. has_math xet RIENG tung cau -> "Bộ Y tế" o cau
    khong-toan khong bi doi Y, du cau khac trong van ban co cong thuc."""
    has_math = bool(_MATH_CONTEXT.search(t))  # xet tren cau GOC (operator con la ky hieu)

    # 1) Phan so: so / so  -> "so phần so"
    t = re.sub(r"(\d)\s*/\s*(\d)", r"\1 phần \2", t)

    # 2) Luy thua dang ^n
    def _pow(m):
        n = m.group(1)
        if len(n) == 1:
            return f" mũ {_POW_WORDS[int(n)]} "
        return f" mũ {n} "
    t = re.sub(r"\^\s*(\d+)", _pow, t)

    # 3) Chi so tren unicode (x² -> x mũ hai)
    for k, v in SUPERSCRIPT.items():
        t = t.replace(k, v)

    # 4) Dau '-' -> "trừ" CHI khi co CHU SO it nhat 1 ben (phep tru thuc su),
    #    tranh bien ten ghep "Griffin-Lim","HiFi-GAN" -> "griffin trừ lim".
    t = re.sub(r"(?<=\d)\s*-\s*(?=[\w(\[√])", " trừ ", t)   # so trừ (so/bien/ngoac)
    t = re.sub(r"(?<=[\w)\]])\s*-\s*(?=\d)", " trừ ", t)   # (so/bien/ngoac) trừ so

    # 5) Ky hieu toan con lai
    for k, v in MATH_SYMBOLS.items():
        t = t.replace(k, v)

    # 6) Ngoac nhon/vuong -> bo
    t = re.sub(r"[{}\[\]]", " ", t)

    # 7) Bien so: chu cai don le -> ten chu cai. CHI khi cau co ngu canh toan.
    if has_math:
        def _letter(m):
            return " " + LETTER_NAMES.get(m.group(0).lower(), m.group(0)) + " "
        t = re.sub(r"(?<![A-Za-zÀ-ỹ])[A-Za-z](?![A-Za-zÀ-ỹ])", _letter, t)
    else:
        # 7b) Cong thuc viet BANG CHU (khong co ky hieu +,^,=): chu cai don DINH KE
        #     tu toan (x mũ, nhân x, x bằng...) van phai doc ten chu cai (x->ích),
        #     tranh F5 doc chu 'x' thanh am "sờ".
        _MWL = "mũ|cộng|trừ|nhân|chia|căn|phần|bằng|phẩy"   # tu toan DUNG SAU chu cai
        _MWR = "mũ|cộng|trừ|nhân|chia|căn"                  # tu toan DUNG TRUOC chu cai (an toan hon)
        t = re.sub(rf"(?<![A-Za-zÀ-ỹ])([A-Za-z])(?=\s+(?:{_MWL})\b)",
                   lambda m: " " + LETTER_NAMES.get(m.group(1).lower(), m.group(1)) + " ", t)
        t = re.sub(rf"\b({_MWR})\s+([A-Za-z])(?![A-Za-zÀ-ỹ])",
                   lambda m: m.group(1) + " " + LETTER_NAMES.get(m.group(2).lower(), m.group(2)) + " ", t)
    return t


def normalize_math(text: str) -> str:
    # Tach theo CAU (. ! ? … + xuong dong), giu dau phan cach, xu ly tung cau rieng.
    parts = re.split(r"([.!?…\n]+)", text)
    t = "".join(p if i % 2 else _normalize_clause(p) for i, p in enumerate(parts))

    # gom khoang trang
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Chuoi de test")
    args = ap.parse_args()
    print("RAW :", args.text)
    print("MATH:", normalize_math(args.text))


if __name__ == "__main__":
    main()
