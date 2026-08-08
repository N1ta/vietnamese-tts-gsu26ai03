#!/usr/bin/env python3
"""
web_f5.py - Web tong hop giong noi tieng Viet (Vietnamese TTS) bang F5-TTS.
Nhap van ban -> chon giong Nam/Nu -> nghe (giong ban xu, nhan tu dataset tu thu).
Xu ly ca van ban dai (tu dong cat doan + ghep). Chuan hoa so/ngay/viet tat san.

CHAY TRONG f5_env:
    C:\\Users\\Nita\\miniconda3\\envs\\f5_env\\python.exe web_f5.py
Mo: http://127.0.0.1:7860  (+ link *.gradio.live neu bat share)
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import gradio as gr
from vi_normalize import normalize, split_for_pause
from gen_baidai_robust import make_units, merge_short
from f5_tts.api import F5TTS

MODEL_DIR = "f5_vi_model"
SR = 24000
SIL = {"long": 0.30, "short": 0.20, "none": 0.10}

VOICES = {
    "Giọng nữ": dict(
        ref="dataset/WOMEN_dataset/wavs/0022.wav",
        ref_text="Trên tất cả các thanh gỗ đều bị cào suốt, không thanh nào còn nguyên vẹn.",
        speed=0.85),
    "Giọng nam": dict(
        ref="dataset/MEN_dataset/wavs/0006.wav",
        ref_text="Sẽ ra sao nếu nhân sự của họ không bị giới hạn bởi thời gian, "
                 "và sẽ ra sao nếu cái chết không còn là sự dễ thoát.",
        speed=1.1),
}

print("[load] Dang tai mo hinh F5-TTS Vietnamese (~1 phut)...")
f5 = F5TTS(model="F5TTS_Base",
           ckpt_file=os.path.join(MODEL_DIR, "model_last.pt"),
           vocab_file=os.path.join(MODEL_DIR, "config.json"))
print("[ok] San sang. Mo trinh duyet o dia chi ben duoi.")


def synth(text, voice, nfe, progress=gr.Progress()):
    if not text or not text.strip():
        raise gr.Error("Vui lòng nhập văn bản tiếng Việt.")
    v = VOICES[voice]
    units = merge_short(make_units(split_for_pause(normalize(text)), 200),
                        minlen=28, maxlen=200)
    pieces = []
    for i, (u, kind) in enumerate(units):
        progress(i / max(1, len(units)), desc=f"Đang tổng hợp đoạn {i + 1}/{len(units)}")
        wav, sr, _ = f5.infer(ref_file=v["ref"], ref_text=v["ref_text"], gen_text=u,
                              remove_silence=True, speed=v["speed"], nfe_step=int(nfe))
        wav = np.asarray(wav, dtype=np.float32)
        wav = wav / (np.abs(wav).max() + 1e-9) * 0.95
        pieces.append(wav)
        pieces.append(np.zeros(int(SIL.get(kind, 0.12) * sr), dtype=np.float32))
    audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    return (SR, audio)


CSS = """
.gradio-container { max-width: 1000px !important; margin: auto; }
#app-header { text-align: center; padding: 18px 0 6px; }
#app-header h1 { font-weight: 700; letter-spacing: -0.01em; margin: 0 0 6px; }
#app-sub { text-align: center; color: #64748b; font-size: 0.95rem; margin: 0 0 2px; }
#app-meta { text-align: center; color: #94a3b8; font-size: 0.85rem; margin: 0; }
#app-footer { text-align: center; color: #94a3b8; font-size: 0.82rem;
              margin-top: 18px; padding-top: 12px; border-top: 1px solid #e2e8f0; }
footer { display: none !important; }
"""

theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

with gr.Blocks(title="VietVoice") as demo:
    gr.HTML("""
    <div id="app-header">
      <h1>VietVoice</h1>
      <p id="app-sub">Chuyển văn bản tiếng Việt thành giọng nói tự nhiên</p>
    </div>
    """)
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            inp = gr.Textbox(
                label="Văn bản đầu vào",
                lines=7,
                placeholder="Nhập văn bản tiếng Việt (câu ngắn hoặc cả đoạn dài). "
                            "Hệ thống tự đọc đúng số, ngày tháng, viết tắt và công thức.")
            with gr.Row():
                voice = gr.Radio(list(VOICES), value="Giọng nữ", label="Chọn giọng", scale=2)
                nfe = gr.Slider(16, 48, value=32, step=8, label="Chất lượng (nhanh ↔ mượt)", scale=3)
            btn = gr.Button("Tổng hợp giọng nói", variant="primary", size="lg")
        with gr.Column(scale=2):
            out = gr.Audio(label="Kết quả", type="numpy")
    gr.Examples(
        examples=[
            ["Xin chào, đây là hệ thống tổng hợp giọng nói tiếng Việt do nhóm chúng tôi xây dựng.", "Giọng nữ", 32],
            ["Hôm nay là ngày 30/4/2025, nhiệt độ ngoài trời khoảng 32 độ, độ ẩm 75%.", "Giọng nam", 32],
            ["Công nghệ AI và ML giúp mô hình đạt độ chính xác lên tới 93,5%.", "Giọng nữ", 40],
        ],
        inputs=[inp, voice, nfe],
        label="Ví dụ mẫu",
    )
    gr.HTML("""
    <div id="app-footer">
      Hỗ trợ số, ngày tháng, viết tắt và văn bản dài · Giọng nam và nữ
    </div>
    """)
    btn.click(synth, [inp, voice, nfe], out)

if __name__ == "__main__":
    # share=False: khong dung gradio.live (frpc loi tren Windows).
    # De co link cong khai, chay them cloudflared (xem start_web_public.bat).
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860,
                inbrowser=True, theme=theme, css=CSS)
