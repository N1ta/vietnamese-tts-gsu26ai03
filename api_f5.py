#!/usr/bin/env python3
"""
api_f5.py - Backend API cho VietVoice (Cach B: frontend Vercel goi backend nay).
Chay trong f5_env, expose qua ngrok domain co dinh.
    C:\\Users\\Nita\\miniconda3\\envs\\f5_env\\python.exe api_f5.py
Endpoint:
    GET  /            -> {"status": "ready"}
    POST /synthesize  -> {text, voice, nfe} -> tra ve audio/wav
"""
import sys, os, io, json, urllib.request
sys.path.insert(0, os.getcwd())
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vi_normalize import normalize, split_for_pause
from gen_baidai_robust import make_units, merge_short
from f5_tts.api import F5TTS

MODEL_DIR = "f5_vi_model"
SR = 24000
SIL = {"long": 0.30, "short": 0.20, "none": 0.10}
VOICES = {
    "nu": dict(ref="dataset/WOMEN_dataset/wavs/0022.wav",
               ref_text="Trên tất cả các thanh gỗ đều bị cào suốt, không thanh nào còn nguyên vẹn.",
               speed=0.85),
    "nam": dict(ref="dataset/MEN_dataset/wavs/0006.wav",
                ref_text="Sẽ ra sao nếu nhân sự của họ không bị giới hạn bởi thời gian, "
                         "và sẽ ra sao nếu cái chết không còn là sự dễ thoát.",
                speed=1.1),
}

# Backend Coqui (VITS / FastSpeech2 / Tacotron2) chay o env `coqui`, port rieng 7861.
# api_f5 chi chuyen tiep; nguoi dung ngoai chi thay 1 ngrok (7860).
COQUI_URL = os.environ.get("COQUI_URL", "http://127.0.0.1:7861/synthesize")
COQUI_MODELS = {"vits", "fs2", "taco"}

print("[load] Dang tai F5-TTS Vietnamese (~1 phut)...")
f5 = F5TTS(model="F5TTS_Base",
           ckpt_file=os.path.join(MODEL_DIR, "model_last.pt"),
           vocab_file=os.path.join(MODEL_DIR, "config.json"))
print("[ok] Backend san sang.")

app = FastAPI(title="VietVoice API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class SynthReq(BaseModel):
    text: str
    model: str = "f5_nu"        # f5_nu | f5_nam | vits | fs2 | taco
    nfe: int = 32
    voice: str = ""             # tuong thich nguoc voi client cu


def proxy_coqui(text, model):
    """Chuyen tiep yeu cau VITS/FS2/Taco sang backend Coqui (env `coqui`, 7861)."""
    payload = json.dumps({"text": text, "model": model}).encode("utf-8")
    req = urllib.request.Request(COQUI_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


@app.get("/")
def root():
    return {"status": "ready", "service": "VietVoice API",
            "models": ["f5_nu", "f5_nam", "vits", "fs2", "taco"]}


@app.post("/synthesize")
def synthesize(r: SynthReq):
    model = (r.model or "f5_nu").lower()
    if r.voice and model in ("f5_nu", "f5_nam"):        # client cu chi gui 'voice'
        model = "f5_nam" if r.voice == "nam" else "f5_nu"

    # --- VITS / FastSpeech2 / Tacotron2: chuyen tiep sang backend Coqui (7861) ---
    if model in COQUI_MODELS:
        try:
            data = proxy_coqui(r.text, model)
        except Exception as e:
            return Response(
                content=(f"Backend Coqui (VITS/FS2/Tacotron2) chua chay hoac loi: {e}. "
                         "Chay start_backend_fixed.bat de bat ca 2 backend.").encode("utf-8"),
                media_type="text/plain", status_code=503)
        return Response(content=data, media_type="audio/wav")

    # --- F5-TTS (Nu / Nam) chay tai cho tren GPU ---
    v = VOICES["nam"] if model == "f5_nam" else VOICES["nu"]
    units = merge_short(make_units(split_for_pause(normalize(r.text)), 200), 28, 200)
    pieces = []
    for u, kind in units:
        wav, sr, _ = f5.infer(ref_file=v["ref"], ref_text=v["ref_text"], gen_text=u,
                              remove_silence=True, speed=v["speed"], nfe_step=int(r.nfe))
        wav = np.asarray(wav, dtype=np.float32)
        wav = wav / (np.abs(wav).max() + 1e-9) * 0.95
        pieces.append(wav)
        pieces.append(np.zeros(int(SIL.get(kind, 0.12) * sr), dtype=np.float32))
    audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)
