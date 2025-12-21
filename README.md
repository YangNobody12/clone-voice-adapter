# Clone Voice Adapter

Repository นี้รวบรวมเครื่องมือสำหรับ Fine-tune โมเดล Orpheus-3B (โมเดลสร้างเสียงด้วย SNAC) ให้เป็นเสียงของคุณเอง

## การติดตั้ง (Installation)

```bash
pip install -r requirements.txt
```

## การใช้งาน (Unified GUI) - แนะนำ!

วิธีที่ง่ายที่สุดคือใช้ Interface แบบ "No-Code" ที่รวมทุกขั้นตอนไว้ในที่เดียว

```bash
python app_gui.py
```

เมื่อรันคำสั่งนี้ โปรแกรมจะเปิดหน้าเว็บที่มี 4 แท็บให้ใช้งาน:
1. **Recog (Verify)**: เครื่องมือสำหรับตรวจสอบความเหมือนของเสียง (Speaker Similarity) เพื่อเช็คว่าเสียงอ้างอิงตรงกับเป้าหมายหรือไม่
2. **Prepare**: เอาไว้สร้าง Dataset (รองรับ ASR ถอดความเสียง, ใช้ Typhoon API แก้คำผิดภาษาไทย, และกรองเสียงพูดที่ไม่ใช่เป้าหมายออก)
3. **Train**: สำหรับสั่ง Fine-tune โมเดลด้วย Dataset ที่เตรียมไว้
4. **Inference**: เครื่องมือสร้างเสียงจากข้อความ (Text-to-Speech) หลังเทรนเสร็จ

## การใช้งานบน Google Colab

เพียงแค่อัปโหลดไฟล์ `fine_tune_clone_voice.ipynb` ขึ้นไปบน Google Colab คุณก็สามารถรันโปรเจกต์นี้บน Cloud ได้ฟรี (โดยใช้ T4 GPU) สคริปต์ใน Notebook จะติดตั้งทุกอย่างและเปิด GUI ให้ใช้งานได้ทันที

## การใช้งานแบบขั้นสูง (Command Line)

หากคุณถนัดใช้ Command Line สามารถเรียกใช้สคริปต์แยกตามฟังก์ชั่นได้ดังนี้:

### 1. เตรียม Dataset (Prepare)
```bash
python recog_voice_for_prepare_data.py
```

### 2. เทรนโมเดล (Training)
```bash
python train.py --metadata path/to/dataset/metadata.csv
```

### 3. สร้างเสียง (Inference)
```bash
python inference.py --text "สวัสดีครับ ผมชื่อสมชาย" --model_path outputs/checkpoint-60
```