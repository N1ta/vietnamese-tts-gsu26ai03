#!/usr/bin/env python3
"""
build_abbr.py — Gộp dữ liệu viết tắt MỚI (New/data_viet_tat_1.json, schema tiếng Việt)
vào bộ cũ dataset/data_viet_tat.json (schema mình), CÓ LỌC AN TOÀN.

Quy tắc:
  - GIỮ NGUYÊN toàn bộ record cũ (giữ benchmark test 93,3%).
  - Thêm từ mới (chưa có trong bộ cũ) theo 2 rổ:
      * doc_chu_cai (CURATED_LETTERS): AI, DL, ML... -> code đọc CHỮ CÁI (ây ai),
        KHÔNG kèm câu test (tránh đối chiếu với bản dịch).
      * còn lại: dùng BẢN DỊCH "đầy đủ" (UBND, WHO, bộ ngành, đơn vị đo...).
  - LỌC AN TOÀN: bỏ từ <=2 ký tự (trừ curated & đơn vị đo >=2), bỏ đơn vị 1 ký tự
    (m/g/l quá dễ trùng). Chat/MXH -> loai chat_slang (mặc định loại khi chạy).

Chạy: python build_abbr.py
"""
import json, os, shutil, sys, collections
sys.stdout.reconfigure(encoding="utf-8")

NEW = "New/data_viet_tat_1.json"
OLD = "dataset/data_viet_tat.json"

# Danh sách ĐỌC CHỮ CÁI (kiểu tiếng Anh: AI->ây ai). Sửa ở đây nếu cần thêm/bớt.
CURATED_LETTERS = {
    "AI", "ML", "DL", "IT", "IOT", "CEO", "CTO", "CFO", "CPU", "GPU", "RAM",
    "USB", "GPS", "API", "SDK", "SQL", "HTML", "CSS", "PDF", "ATM", "SIM",
    "ID", "IP", "UI", "UX", "HR", "PR", "OS", "PC", "TV", "VPN", "URL", "HTTP",
}
UNIT_CAT = "Đơn vị đo lường"
SLANG_CAT = "Chat/MXH"

old = json.load(open(OLD, encoding="utf-8"))
old_tus = {str(r.get("tu_viet_tat", "")).strip() for r in old}
out = list(old)  # giữ nguyên cũ

new = json.load(open(NEW, encoding="utf-8"))
stats = collections.Counter()
seen = set(old_tus)
for r in new:
    tu = str(r.get("từ viết tắt", "")).strip()
    full = str(r.get("đầy đủ", "")).strip()
    cat = str(r.get("Loại", "")).strip()
    goc = str(r.get("câu gốc", "")).strip()
    norm = str(r.get("câu normalize", "")).strip()
    if not tu or not full or tu in seen:
        stats["skip_dup_or_empty"] += 1
        continue

    rec = {"tu_viet_tat": tu, "day_du": full}
    if tu in CURATED_LETTERS:
        rec["loai"] = "doc_chu_cai"           # đọc chữ cái, KHÔNG câu test
        stats["letters"] += 1
    elif cat == UNIT_CAT:                       # đơn vị đo (viết thường hợp lệ: km, kg)
        if len(tu) < 2:                        # bỏ đơn vị 1 ký tự (m, g, l)
            stats["drop_unit_1char"] += 1
            continue
        rec.update(loai="don_vi", cau_goc=goc, cau_normalize=norm)
        stats["unit"] += 1
    elif not tu.isupper():                      # KEY KHÔNG IN HOA = từ thường trá hình (ban, vs...) -> BỎ
        stats["drop_lowercase"] += 1
        continue
    elif cat == SLANG_CAT:
        rec.update(loai="chat_slang", cau_goc=goc, cau_normalize=norm)
        stats["slang"] += 1
    elif len(tu) <= 2:                          # bỏ viết tắt IN HOA ngắn rủi ro (EU, UN...)
        stats["drop_short"] += 1
        continue
    else:
        rec.update(loai="viet_tat", cau_goc=goc, cau_normalize=norm)
        stats["translation"] += 1
    seen.add(tu)
    out.append(rec)

shutil.copy(OLD, "dataset/data_viet_tat.backup2.json")
json.dump(out, open(OLD, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("Record cũ:", len(old), "| thêm mới:", len(out) - len(old), "| tổng:", len(out))
print("Chi tiết:", dict(stats))
