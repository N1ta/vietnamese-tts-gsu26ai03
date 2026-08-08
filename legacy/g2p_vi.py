#!/usr/bin/env python3
"""
g2p_vi.py
=========
Chuyen van ban DA NORMALIZE thanh token (char hoac phoneme) cho TTS, roi:
  - sinh filelist moi voi cot token,
  - dem vocab THAT,
  - ghi de vocab/n_symbols/n_vocab vao ca 3 config.json (cho cong bang so sanh).

CHE DO (--mode):
  char     : tach ky tu, dung symbols.py co san. KHONG can lexicon.
  phoneme  : G2P. Tra LEXICON nho truoc (tu bat quy tac), phan con lai dung
             quy tac (thu vien viphoneme neu co; neu khong, fallback rule don
             gian de pipeline van chay duoc).

LEXICON (--lexicon lexicon_vi.txt) - chi can o mode phoneme:
  Moi dong:  tu<TAB>phoneme1 phoneme2 ...
  Vi du:     wifi    w ai f ai
             microsoft   m a i k r o s o f t
  Dung de VA cac tu doc bat quy tac (ten rieng, tu muon). KHONG can liet ke
  het moi tu - tu thuan Viet de quy tac tu xu ly.

Cach dung:
    # mode char (don gian nhat, dung symbols.py)
    python g2p_vi.py --mode char \
        --in  dataset/single_speaker/VIVOSSPK35/metadata_norm.csv \
        --out dataset/single_speaker/VIVOSSPK35/metadata_char.csv \
        --symbols symbols.py --models-dir models

    # mode phoneme (lai: lexicon + quy tac)
    python g2p_vi.py --mode phoneme \
        --in  dataset/single_speaker/VIVOSSPK35/metadata_norm.csv \
        --out dataset/single_speaker/VIVOSSPK35/metadata_phone.csv \
        --lexicon lexicon_vi.txt --models-dir models
"""

import argparse
import json
import re
from pathlib import Path


# --------------------------------------------------------------------------- #
# symbols.py loader (mode char)
# --------------------------------------------------------------------------- #
def load_symbols(symbols_path):
    """Doc list `symbols` tu file symbols.py mot cach AN TOAN (khong exec).

    File symbols.py cua nhom co the chua dong hong (vd: phan tu '\\' bi escape
    sai -> lam exec/regex toan file thi loi). Ta parse TUNG DONG: moi dong lay
    1 ky tu nam giua cap dau nhay dau tien, bo qua dong hong.
    """
    syms, seen = [], set()
    for raw in Path(symbols_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip().rstrip(",")
        if not (len(line) >= 2 and line[0] in "\"'"):
            continue
        q = line[0]
        end = line.find(q, 1)
        if end == -1:
            continue
        inner = line[1:end]
        # giai ma escape pho bien
        inner = (inner.replace("\\\\", "\\").replace('\\"', '"')
                      .replace("\\'", "'").replace("\\n", "\n")
                      .replace("\\t", "\t"))
        if inner == "":
            continue
        # phong truong hop dong hong con sot escape la: chi giu 1 ky tu dau
        ch = inner[0]
        if ch not in seen:
            seen.add(ch); syms.append(ch)
    if not syms:
        raise SystemExit("Khong trich duoc `symbols` tu " + symbols_path)
    return syms


# --------------------------------------------------------------------------- #
# Lexicon loader (mode phoneme)
# --------------------------------------------------------------------------- #
def load_lexicon(path):
    lex = {}
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) == 2:
                lex[parts[0].lower()] = parts[1].split()
    return lex


# --------------------------------------------------------------------------- #
# Phoneme backend
# --------------------------------------------------------------------------- #
def get_phonemizer():
    """Tra ve ham word->[phonemes]. Uu tien viphoneme, fallback rule don gian."""
    try:
        from viphoneme import vi2IPA_split  # type: ignore

        def ph(word):
            ipa = vi2IPA_split(word, "/")
            return [p for p in ipa.split("/") if p]
        return ph, "viphoneme"
    except Exception:
        # Fallback: tach am dau / van / thanh rat tho so. Du de pipeline chay,
        # KHONG dat chat luong nghien cuu - cai viphoneme de dung that.
        def ph(word):
            return list(word)  # degrade ve ky tu
        return ph, "fallback-char"


def g2p_word(word, lex, phon):
    w = word.lower()
    if w in lex:
        return lex[w]            # bat quy tac -> tra lexicon
    return phon(w)               # con lai -> quy tac


# --------------------------------------------------------------------------- #
def tokenize_line(text, mode, symset, lex, phon):
    if mode == "char":
        toks = [c for c in text if c in symset or c == " "]
        return toks
    # phoneme
    out = []
    for word in text.split():
        out.extend(g2p_word(word, lex, phon))
        out.append(" ")
    if out and out[-1] == " ":
        out.pop()
    return out


def update_configs(models_dir, vocab_size, mode):
    mdir = Path(models_dir)
    targets = {
        "tacotron2": "n_symbols",
        "FastSpeech2_vi": "vocab_size",
        "vits": ("model", "n_vocab"),
    }
    for name, key in targets.items():
        cfg_path = mdir / name / "config.json"
        if not cfg_path.exists():
            print(f"  [warn] khong thay {cfg_path}, bo qua")
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(key, tuple):
            cfg.setdefault(key[0], {})[key[1]] = vocab_size
        else:
            cfg[key] = vocab_size
        cfg["_vocab_info"] = {"mode": mode, "vocab_size": vocab_size}
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"  [ok] {name}: vocab -> {vocab_size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["char", "phoneme"], default="phoneme")
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument("--symbols", default=None, help="symbols.py (mode char).")
    ap.add_argument("--lexicon", default=None, help="lexicon_vi.txt (mode phoneme).")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    symset, lex, phon, backend = set(), {}, None, None
    if args.mode == "char":
        if not args.symbols:
            ap.error("mode char can --symbols symbols.py")
        symset = set(load_symbols(args.symbols))
        print(f"[init] char mode, {len(symset)} symbols")
    else:
        lex = load_lexicon(args.lexicon)
        phon, backend = get_phonemizer()
        print(f"[init] phoneme mode, backend={backend}, lexicon={len(lex)} tu")
        if backend == "fallback-char":
            print("  [warn] chua cai viphoneme -> dang degrade ve ky tu. "
                  "Cai bang: pip install viphoneme")

    vocab = set()
    out_lines, n_oov, total_words = [], 0, 0
    for line in Path(args.infile).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if parts[0].endswith(".wav"):
            wav, text = parts[0], "|".join(parts[1:])
        else:
            wav, text = None, parts[-1]
        toks = tokenize_line(text, args.mode, symset, lex, phon)
        vocab.update(t for t in toks if t != " ")
        if args.mode == "phoneme":
            for w in text.split():
                total_words += 1
                if w.lower() not in lex:
                    n_oov += 1
        tok_str = " ".join("<sp>" if t == " " else t for t in toks)
        if wav:
            out_lines.append(f"{wav}|{tok_str}")
        else:
            parts[2] = "{" + tok_str + "}"
            out_lines.append("|".join(parts))

    Path(args.outfile).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    # vocab + token dac biet (pad, eos, space)
    vocab_size = len(vocab) + 3
    print(f"[done] {len(out_lines)} dong -> {args.outfile}")
    print(f"[vocab] {len(vocab)} token rieng (+3 dac biet) = {vocab_size}")
    if args.mode == "phoneme":
        cov = 100 * (1 - n_oov / max(total_words, 1))
        print(f"[lexicon] do phu lexicon: {cov:.1f}% tu khop "
              f"(con lai do quy tac xu ly)")
    print("[config] cap nhat vocab vao 3 model:")
    update_configs(args.models_dir, vocab_size, args.mode)
    print("\nLuu y: doi vocab -> so tham so doi chut. Chay lai make_configs.py "
          "voi --vocab {} de tinh chinh ve ~50M neu can.".format(vocab_size))


if __name__ == "__main__":
    main()
