# Vietnamese Speech Synthesis Using Deep Learning Models

**Capstone Project — GSU26AI03 · FPT University**
Tổng hợp giọng nói tiếng Việt bằng các mô hình học sâu.

> **Supervisor:** MSc. Đỗ Đức Hào
> **Team:** Nguyễn Đức Hiển (SE173053) · Lưu Đức Anh (SE183141) · Nguyễn Trần Trung Tín (SE183050)

---

## Giới thiệu

Dự án xây dựng một hệ thống **Text-to-Speech (TTS)** tiếng Việt hoàn chỉnh và **so sánh hai hướng tiếp cận**:

- **Nhánh A — từ đầu (from-scratch):** huấn luyện 3 kiến trúc **Tacotron2, FastSpeech2, VITS** + vocoder **HiFi-GAN** trên cùng bộ dữ liệu **InFoRe** (16 kHz), để so sánh kiến trúc một cách công bằng.
- **Nhánh B — pretrained:** dùng mô hình **F5-TTS** + **voice cloning zero-shot** trên **dataset tự thu** (2 giọng nam/nữ, 24 kHz) → giọng bản xứ chất lượng cao làm sản phẩm demo.

Tất cả dùng chung một **front-end** (chuẩn hóa văn bản + cắt đoạn văn bản dài) và được đánh giá bằng **MOS** (26 người nghe, chấm mù).

## Kết quả chính (MOS)

| Hệ thống | Loại | SR | MOS |
|---|---|---|---|
| **F5-TTS (Nữ)** | pretrained | 24 kHz | **4.42 ± 0.47** (cao nhất) |
| F5-TTS (Nam) | pretrained | 24 kHz | 3.89 ± 1.10 |
| VITS | from-scratch, end-to-end | 16 kHz | 3.17 ± 1.01 |
| FastSpeech2 + HiFi-GAN | from-scratch | 16 kHz | 3.13 ± 1.04 |
| Tacotron2 + HiFi-GAN | from-scratch | 16 kHz | 2.72 ± 1.13 |

Độ tin cậy Cronbach's α = 0.96; ANOVA F(4,100) = 34.52, p < .001.

## Cấu trúc mã nguồn

| File | Vai trò |
|---|---|
| `vi_normalize.py`, `math_normalize.py` | Chuẩn hóa văn bản (số, ngày, viết tắt, công thức) |
| `align_punct.py` | Khôi phục dấu câu bằng forced alignment |
| `data_audit.py`, `test_normalize.py` | Kiểm tra dữ liệu / độ chính xác chuẩn hóa |
| `demo_vits.py`, `demo_tacotron.py` | Sinh demo nhánh A (VITS / Tacotron2 / FastSpeech2 + HiFi-GAN) |
| `demo_f5.py`, `gen_baidai_robust.py` | Sinh giọng F5 (câu ngắn / văn bản dài, best-of-N) |
| `web_f5.py` | Ứng dụng web (Gradio) — nhập text → sinh audio |

> **Lưu ý:** các checkpoint mô hình, dữ liệu audio và file .wav **không** được lưu trên GitHub (dung lượng lớn). Dataset InFoRe công khai tại HuggingFace `ntt123/infore`; mô hình F5-TTS Vietnamese: `hynt/F5-TTS-Vietnamese-ViVoice`.

## Chạy thử

Hai môi trường conda tách biệt (do xung đột thư viện `transformers`):

- **`coqui`** (nhánh A — Coqui-TTS 0.27.5): chạy `demo_vits.py`, `demo_tacotron.py`.
- **`f5_env`** (nhánh B — F5-TTS): chạy `web_f5.py` (web demo), `demo_f5.py`.

```bash
# Web demo (trong f5_env):
python web_f5.py
```

## Công nghệ

Python 3.10 · PyTorch (CUDA 12.1) · Coqui-TTS 0.27.5 · F5-TTS · HiFi-GAN · Gradio · wav2vec2 (forced alignment) · Demucs.

---

*Đồ án tốt nghiệp — FPT University, 2026.*
