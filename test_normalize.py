#!/usr/bin/env python3
"""
test_normalize.py
=================
Do DO CHINH XAC chuan hoa van ban tren bo 1200 cau cua nhom
(dataset/data_viet_tat.json: cau_goc -> cau_normalize).

Cach do:
  - Voi moi cap, so sanh normalize(cau_goc) vs normalize(cau_normalize).
    (Ca hai qua CUNG pipeline nen khac biet hoa/dau cau bi triet tieu;
     chi con khac biet o cach bung viet tat -> dung de do chinh xac.)
  - Bao cao ti le dung TONG THE va theo TUNG NHOM (loai).

Cach dung:
    python test_normalize.py
    python test_normalize.py --slang        # tinh ca nhom chat_slang
    python test_normalize.py --show-fail 10 # in 10 cau sai de soi
"""

import argparse
import json
from collections import defaultdict

from vi_normalize import normalize, load_abbr

JSON = "dataset/data_viet_tat.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=JSON)
    ap.add_argument("--slang", action="store_true", help="Tinh ca nhom chat_slang")
    ap.add_argument("--show-fail", type=int, default=5, help="So cau sai in ra moi nhom")
    args = ap.parse_args()

    data = json.load(open(args.json, encoding="utf-8"))
    abbr = load_abbr(include_slang=args.slang)

    by_loai = defaultdict(lambda: {"pass": 0, "total": 0, "fails": []})
    overall_pass = overall_total = 0

    for x in data:
        goc = str(x.get("cau_goc", "")).strip()
        norm_expected = str(x.get("cau_normalize", "")).strip()
        loai = x.get("loai", "?")
        if not goc or not norm_expected:
            continue  # bo record khong co cau vi du (vd cac tu moi them)
        if loai == "chat_slang" and not args.slang:
            continue

        got = normalize(goc, abbr)
        want = normalize(norm_expected, abbr)
        ok = (got == want)

        by_loai[loai]["total"] += 1
        overall_total += 1
        if ok:
            by_loai[loai]["pass"] += 1
            overall_pass += 1
        elif len(by_loai[loai]["fails"]) < args.show_fail:
            by_loai[loai]["fails"].append((x.get("tu_viet_tat", ""), got, want))

    print("=" * 64)
    print(f"{'NHOM (loai)':<16} {'Dung':>6} {'Tong':>6} {'Ti le':>8}")
    print("-" * 64)
    for loai in sorted(by_loai):
        d = by_loai[loai]
        pct = d["pass"] * 100 / d["total"] if d["total"] else 0
        print(f"{loai:<16} {d['pass']:>6} {d['total']:>6} {pct:>7.1f}%")
    print("-" * 64)
    pct = overall_pass * 100 / overall_total if overall_total else 0
    print(f"{'TONG THE':<16} {overall_pass:>6} {overall_total:>6} {pct:>7.1f}%")
    print("=" * 64)

    # In vai cau sai de soi
    for loai in sorted(by_loai):
        fails = by_loai[loai]["fails"]
        if not fails:
            continue
        print(f"\n[SAI] nhom {loai} (toi da {args.show_fail} vi du):")
        for tu, got, want in fails:
            print(f"  ({tu}) got : {got[:70]}")
            print(f"       want: {want[:70]}")


if __name__ == "__main__":
    main()
