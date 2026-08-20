#!/usr/bin/env python3
"""
api_coqui.py - Backend Coqui cho VietVoice (VITS / FastSpeech2 / Tacotron2).
Chay trong env `coqui`, lang nghe cuc bo o 127.0.0.1:7861.
api_f5.py (env f5_env, cong 7860) se CHUYEN TIEP cac model nay sang day,
nen nguoi dung ben ngoai chi thay 1 ngrok duy nhat.

    C:\\Users\\Nita\\miniconda3\\envs\\coqui\\python.exe api_coqui.py

Endpoint:
    GET  /            -> {"status": "ready", "models": [...]}
    POST /synthesize  -> {text, model}  (model = vits|fs2|taco) -> audio/wav

Ghi chu: mac dinh chay CPU (COQUI_DEVICE=cpu) de KHONG tranh VRAM voi F5 (6GB).
Doi sang GPU: dat bien moi truong COQUI_DEVICE=cuda truoc khi chay.
"""
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

sys.path.insert(0, os.getcwd())
from vi_normalize import normalize, split_for_pause, strip_punct

DEVICE = os.environ.get("COQUI_DEVICE", "cpu").lower()
PORT = int(os.environ.get("COQUI_PORT", "7861"))
SIL = {"long": 0.30, "short": 0.20, "none": 0.10}

# --- Checkpoint (dung dung path cac script demo_*.py da kiem chung) ---
VITS_RUN = "models/vits_punct/runs/vits_infore_punct-June-21-2026_12+15AM-0000000"
TACO_RUN = "models/tacotron2/runs/tacotron2_infore_punct-June-26-2026_08+39PM-0000000"
FS2_RUN = "models/fastspeech2_punct/runs/fastspeech2_infore-July-25-2026_05+19AM-0000000"
HIFI_RUN = "models/hifigan/runs/hifigan_infore-June-27-2026_07+33PM-0000000"
HIFI_CKPT = f"{HIFI_RUN}/checkpoint_68572.pth"

# Doi tuong da nap + sample rate; None = chua nap / nap loi.
_M = {"vits": None, "taco": None, "fs2": None}
_SR = {"vits": 16000, "taco": 16000, "fs2": 16000}
_LOCK = threading.Lock()


# --------------------------- helper xu ly audio --------------------------- #
def trim_sil(wav, sr, thresh=0.015, pad=0.02):
    if len(wav) == 0:
        return wav
    idx = np.where(np.abs(wav) > thresh)[0]
    if len(idx) == 0:
        return wav
    p = int(pad * sr)
    return wav[max(0, idx[0] - p):min(len(wav), idx[-1] + p)]


def trim_tail(wav, sr, frac=0.22, win_ms=30, pad_ms=60):
    """Cat duoi 'ummm' do Tacotron2 over-generate (giong demo_tacotron.py)."""
    if len(wav) == 0:
        return wav
    win = max(1, int(win_ms / 1000 * sr))
    n = len(wav) // win
    if n < 2:
        return wav
    env = np.array([np.sqrt((wav[i * win:(i + 1) * win] ** 2).mean()) for i in range(n)])
    mx = env.max()
    if mx <= 0:
        return wav
    strong = np.where(env > frac * mx)[0]
    if len(strong) == 0:
        return wav
    end = min(len(wav), (strong[-1] + 1) * win + int(pad_ms / 1000 * sr))
    return wav[:end]


def apply_fade(wav, sr, fade_ms=10):
    n = min(int(fade_ms / 1000 * sr), len(wav) // 2)
    if n > 0:
        wav = wav.copy()
        wav[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
        wav[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return wav


def make_units(segs, max_chars=200):
    """Gom doan (da tach o dau cau) thanh don vi <= max_chars (giong demo_tacotron.py)."""
    units = []
    for text, kind in segs:
        if len(text) <= max_chars:
            units.append((text, kind))
            continue
        words, piece, pieces = text.split(), "", []
        for w in words:
            if len(piece) + len(w) + 1 <= max_chars:
                piece = (piece + " " + w).strip()
            else:
                pieces.append(piece)
                piece = w
        if piece:
            pieces.append(piece)
        for i, p in enumerate(pieces):
            units.append((p, kind if i == len(pieces) - 1 else "none"))
    return units


# ------------------------------ nap model -------------------------------- #
def _load(model):
    """Nap 1 model (lazy, co cache). Tra doi tuong hoac None neu loi."""
    with _LOCK:
        if _M[model] is not None:
            return _M[model]
        try:
            if model == "vits":
                from TTS.api import TTS
                from TTS.config import load_config
                tts = TTS(model_path=f"{VITS_RUN}/best_model.pth",
                          config_path=f"{VITS_RUN}/config.json")
                tts.to(DEVICE)
                _SR["vits"] = load_config(f"{VITS_RUN}/config.json").audio.sample_rate
                _M["vits"] = tts
            else:
                from TTS.utils.synthesizer import Synthesizer
                run = TACO_RUN if model == "taco" else FS2_RUN
                syn = Synthesizer(
                    tts_checkpoint=f"{run}/best_model.pth",
                    tts_config_path=f"{run}/config.json",
                    vocoder_checkpoint=HIFI_CKPT,
                    vocoder_config=f"{HIFI_RUN}/config.json",
                    use_cuda=(DEVICE == "cuda"),
                )
                if model == "taco":                       # chan over-generate
                    try:
                        syn.tts_model.decoder.max_decoder_steps = 400
                    except Exception:
                        pass
                _SR[model] = syn.output_sample_rate
                _M[model] = syn
            print(f"[ok] Da nap model: {model} @ {_SR[model]} Hz ({DEVICE})")
        except Exception as e:
            print(f"[ERR] Nap model {model} that bai: {e}")
            _M[model] = None
        return _M[model]


# ------------------------------ tong hop --------------------------------- #
def _synth_seg(model, seg):
    obj, sr = _M[model], _SR[model]
    if model == "vits":
        buf = io.BytesIO()
        obj.tts_to_file(text=seg, file_path=buf)          # VITS end-to-end
        buf.seek(0)
        wav, _ = sf.read(buf, dtype="float32")
        return apply_fade(trim_sil(wav, sr), sr)
    seg = strip_punct(seg).strip()                        # Taco/FS2: vocab khong co dau cau
    if not seg:
        return np.zeros(0, dtype=np.float32)
    wav = np.asarray(obj.tts(seg, split_sentences=False), dtype=np.float32)
    wav = trim_sil(wav, sr)
    if model == "taco":
        wav = trim_tail(wav, sr)
    return apply_fade(wav, sr)


def synthesize(model, text):
    if model not in _M:
        raise ValueError(f"model khong hop le: {model}")
    if _load(model) is None:
        raise RuntimeError(f"khong nap duoc model {model} (xem log backend Coqui)")
    sr = _SR[model]
    units = make_units(split_for_pause(normalize(text)), 200)
    pieces = []
    for i, (seg, kind) in enumerate(units):
        p = _synth_seg(model, seg)
        if len(p):
            pieces.append(p)
            if i < len(units) - 1:
                pieces.append(np.zeros(int(SIL.get(kind, 0.1) * sr), dtype=np.float32))
    audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    audio = audio / (np.abs(audio).max() + 1e-9) * 0.95
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()


# ------------------------------ HTTP server ------------------------------ #
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, json.dumps({"status": "ready", "models": list(_M)}).encode(),
                   "application/json")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            model = str(req.get("model", "vits")).lower()
            text = req.get("text", "")
            wav = synthesize(model, text)
            self._send(200, wav, "audio/wav")
        except Exception as e:
            self._send(500, f"Coqui loi: {e}".encode("utf-8"), "text/plain")

    def log_message(self, *a):
        pass   # tat log truy cap cho gon


if __name__ == "__main__":
    print(f"[coqui] Backend VITS/FS2/Tacotron2 tren 127.0.0.1:{PORT} (device={DEVICE})")
    print("[coqui] Model nap lazy o lan goi dau tien (Tacotron2 CPU co the cham).")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
