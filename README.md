# 🎙️ Clone Voice Adapter

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YangNobody12/clone-voice-adapter/blob/main/fine_tune_clone_voice.ipynb)

**Fine-tune TTS โมเดลให้เป็นเสียงของคุณเอง ด้วย LoRA + Unsloth**

</div>

---

## ✨ Features

- 🎤 **บันทึกเสียง** → ตัดอัตโนมัติเป็น 11-15 วินาที ลบเสียงเงียบ
- 🗣️ **Typhoon ASR** → Speech-to-Text ภาษาไทยด้วย API
- 📊 **สร้าง Dataset** → Format LJSpeech อัตโนมัติ
- 🏋️ **Train LoRA** → Fine-tune ด้วย Unsloth บน T4 GPU ฟรี
- 🔊 **Inference** → สร้างเสียงจาก Text ด้วยโมเดลที่เทรน

---

## 🚀 Quick Start

### Google Colab (แนะนำ!)
```
กดปุ่ม "Open in Colab" ด้านบน → Run all cells
```

### Local
```bash
pip install -r requirements.txt
python app_gui.py
```

> ⚠️ **ต้องการ:** Python 3.10+, CUDA GPU, FFmpeg

---

## 📋 Pipeline

```
1. บันทึกเสียง (1-5 นาที)
      ↓
2. ตัดเป็น segments 11-15s + ลบเสียงเงียบ
      ↓
3. Typhoon STT → transcribe เป็นข้อความ
      ↓
4. สร้าง metadata.csv (LJSpeech format)
      ↓
5. Train LoRA (60-360 steps)
      ↓
6. Inference → สร้างเสียงจาก text
```

---

## 🖥️ GUI Tabs

| Tab | หน้าที่ |
|-----|---------|
| **1. บันทึก & สร้าง Dataset** | Record/Upload → ตัด → STT → metadata.csv |
| **2. Train Model** | Fine-tune LoRA ด้วย config ที่กำหนด |
| **3. Inference** | Text-to-Speech ด้วยโมเดลที่เทรน |

---

## ⚙️ Configuration

แก้ไขได้ใน `config.py`:

```python
# Model
model_name = "unsloth/orpheus-3b-0.1-ft"  # หรือ custom model
max_seq_length = 2048
r = 64                    # LoRA rank

# Training
max_steps = 360           # จำนวน training steps
learning_rate = 2e-4
```

---

## 📁 Output Structure

```
dataset/
├── metadata.csv         # wavs/xxx.wav|ข้อความ
└── wavs/
    ├── audio_001.wav
    └── audio_002.wav

outputs/
└── checkpoint-360/      # LoRA weights
```

---

## 🔧 Requirements

- **GPU:** NVIDIA with CUDA (T4, V100, A100, RTX)
- **VRAM:** 16GB+ recommended
- **Python:** 3.10+
- **FFmpeg:** สำหรับ audio processing

```bash
# Windows
choco install ffmpeg

# Linux
sudo apt install ffmpeg
```

---

## 📚 Files

| File | Description |
|------|-------------|
| `app_gui.py` | Main Gradio GUI |
| `audio_processor.py` | ตัด audio, ลบเสียงเงียบ |
| `typhoon_stt.py` | Typhoon ASR API |
| `dataset_prep.py` | เตรียม dataset สำหรับ train |
| `train.py` | Training script |
| `inference.py` | Text-to-Speech inference |
| `config.py` | Model & training config |

---

## 🙏 Credits

- [Unsloth](https://github.com/unslothai/unsloth) - Fast LoRA fine-tuning
- [Orpheus-3B](https://huggingface.co/canopylabs/orpheus-3b) - Base TTS model
- [SNAC](https://github.com/hubertsiuzdak/snac) - Audio codec
- [Typhoon](https://opentyphoon.ai) - Thai ASR API

---

## 📄 License

MIT License