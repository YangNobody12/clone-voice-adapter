"""
Clone Voice Adapter Studio - Main GUI
Pipeline: Record → Cut Audio → Typhoon STT → Prepare Data → Train → Inference
"""
import gradio as gr
import os
import tempfile
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
import threading

# Local modules
from audio_processor import process_audio_for_dataset, get_audio_duration
from typhoon_stt import process_wavs_to_metadata, transcribe_all_segments, save_metadata_csv
from train import train
from inference import InferenceEngine

# ===============================
# HISTORY SYSTEM
# ===============================
HISTORY_DIR = Path(__file__).parent / "output_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


# ===============================
# TAB 1: RECORD & CREATE DATASET
# ===============================
def process_recording_pipeline(audio_path, typhoon_key, min_sec, max_sec, progress=gr.Progress()):
    """
    Full pipeline:
    1. ตัด audio เป็น 11-15s segments, ลบเสียงเงียบ
    2. ส่งแต่ละ segment ไป Typhoon STT
    3. สร้าง metadata.csv (LJSpeech format)
    4. สร้าง ZIP file
    """
    if not audio_path:
        return "❌ กรุณาบันทึกหรืออัปโหลดเสียง", "", "", None
    
    if not typhoon_key:
        return "❌ กรุณาใส่ Typhoon API Key", "", "", None
    
    try:
        audio_path = Path(audio_path)
        basename = audio_path.stem
        temp_dir = Path(tempfile.mkdtemp(prefix="voice_clone_"))
        
        # Step 1: Process audio (remove silence, cut segments)
        progress(0.1, desc="กำลังตัดเสียง...")
        print(f"[Pipeline] Step 1: Processing audio...")
        segment_paths = process_audio_for_dataset(
            str(audio_path), 
            str(temp_dir),
            min_sec=int(min_sec),
            max_sec=int(max_sec)
        )
        
        if not segment_paths:
            return "❌ ไม่สามารถตัดเสียงได้", "", "", None
        
        # Calculate total duration
        total_segments = len(segment_paths)
        segments_info = f"ตัดได้ {total_segments} segments"
        
        # Step 2: Transcribe with Typhoon
        progress(0.3, desc="กำลัง transcribe ด้วย Typhoon...")
        print(f"[Pipeline] Step 2: Transcribing {total_segments} segments...")
        
        wavs_dir = temp_dir / "wavs"
        
        def update_progress(current, total, filename):
            progress(0.3 + (0.5 * current / total), desc=f"Transcribing {filename}...")
        
        transcriptions = transcribe_all_segments(
            str(wavs_dir), 
            typhoon_key,
            progress_callback=update_progress
        )
        
        if not transcriptions:
            return "❌ Typhoon ไม่สามารถ transcribe ได้ (ตรวจสอบ API Key)", segments_info, "", None
        
        # Step 3: Save metadata.csv (LJSpeech format)
        progress(0.85, desc="กำลังสร้าง metadata...")
        print(f"[Pipeline] Step 3: Saving metadata...")
        
        metadata_path = temp_dir / "metadata.csv"
        save_metadata_csv(transcriptions, str(metadata_path), format_type="ljspeech")
        
        # Read metadata content for display
        metadata_content = metadata_path.read_text(encoding="utf-8")
        
        # Step 4: Create ZIP
        progress(0.95, desc="กำลังสร้าง ZIP...")
        print(f"[Pipeline] Step 4: Creating ZIP...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = temp_dir / f"{basename}_dataset_{timestamp}.zip"
        
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(metadata_path, "metadata.csv")
            for wav_file in wavs_dir.glob("*.wav"):
                zf.write(wav_file, f"wavs/{wav_file.name}")
        
        # Copy to history
        history_zip = HISTORY_DIR / zip_path.name
        shutil.copy(zip_path, history_zip)
        
        # Also save dataset folder for training
        dataset_dir = Path(__file__).parent / "dataset"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        shutil.copytree(temp_dir, dataset_dir)
        
        progress(1.0, desc="เสร็จสิ้น!")
        
        # Create preview
        preview_lines = []
        for i, (filename, text) in enumerate(transcriptions[:10], 1):
            preview_lines.append(f"[{i}] {text[:60]}...")
        preview = "\n".join(preview_lines)
        
        status = f"✅ สำเร็จ! สร้าง {len(transcriptions)} segments\n📁 Dataset บันทึกที่: {dataset_dir}"
        
        return status, preview, metadata_content, str(zip_path)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", "", "", None


# ===============================
# TAB 2: TRAINING
# ===============================
def trigger_training(metadata_path, max_steps, learning_rate):
    """Start training in background thread"""
    try:
        if not os.path.exists(metadata_path):
            return f"❌ ไม่พบไฟล์ {metadata_path}"
        
        def run_train():
            train(metadata_path, overrides={
                "max_steps": int(max_steps), 
                "learning_rate": float(learning_rate)
            })
        
        t = threading.Thread(target=run_train)
        t.start()
        
        return "🚀 เริ่ม Training แล้ว! ดู progress ใน console..."
        
    except Exception as e:
        return f"❌ Training Error: {str(e)}"


# ===============================
# TAB 3: INFERENCE
# ===============================
inference_engine = None

def run_inference(text, model_path):
    """Generate audio from text"""
    global inference_engine
    
    if not text.strip():
        return None, "❌ กรุณาใส่ข้อความ"
    
    if not os.path.exists(model_path):
        return None, f"❌ ไม่พบ model ที่ {model_path}"
    
    try:
        if inference_engine is None or inference_engine.model_path != model_path:
            print(f"[Inference] Loading model from {model_path}...")
            inference_engine = InferenceEngine(model_path=model_path)
        
        output_path = inference_engine.generate(text)
        return output_path, "✅ สร้างเสียงสำเร็จ!"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Inference Error: {str(e)}"


# ===============================
# GRADIO UI
# ===============================
with gr.Blocks(title="Clone Voice Adapter Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ Clone Voice Adapter Studio
    **Pipeline:** บันทึกเสียง → ตัด 11-15s → Typhoon STT → Train → Inference
    """)

    with gr.Tabs():
        # =====================
        # TAB 1: RECORD & PREPARE
        # =====================
        with gr.Tab("1. 🎤 บันทึก & สร้าง Dataset"):
            gr.Markdown("""
            ### ขั้นตอน
            1. บันทึกเสียง **1-5 นาที** (อ่านบทด้านล่าง)
            2. ใส่ **Typhoon API Key**
            3. กด **สร้าง Dataset**
            """)
            
            # Reading script
            with gr.Accordion("📖 บทอ่านสำหรับบันทึก (กดเพื่อดู)", open=False):
                gr.Markdown("""
                **บทอ่านทดสอบ (อ่านช้าๆ ชัดๆ ประมาณ 2-3 นาที):**
                
                ---
                
                "สวัสดีครับ วันนี้เราจะมาทดสอบระบบสังเคราะห์เสียงภาษาไทย
                ด้วยเทคโนโลยีปัญญาประดิษฐ์ การสร้างเสียงสังเคราะห์ที่มีความเป็นธรรมชาติ
                จำเป็นต้องอาศัยข้อมูลเสียงที่มีคุณภาพและความชัดเจนในการออกเสียง

                ในปัจจุบัน เทคโนโลยี AI ได้เข้ามามีบทบาทสำคัญในชีวิตประจำวัน
                ไม่ว่าจะเป็นผู้ช่วยเสมือนจริง การแปลภาษาอัตโนมัติ หรือการอ่านข่าว
                การที่เราสามารถโคลนเสียงของตนเองได้นั้น จะช่วยเปิดโอกาสใหม่ๆ มากมาย

                ระบบนี้ใช้โมเดล Orpheus ซึ่งเป็นโมเดล text-to-speech ที่ทันสมัย
                ผมกำลังทดสอบการพูดในระดับเสียงปกติ ไม่เร็วและไม่ช้าจนเกินไป
                เพื่อให้ระบบสามารถเรียนรู้ลักษณะเฉพาะของเสียงได้อย่างถูกต้อง

                ขอบคุณที่ร่วมทดสอบระบบครับ"
                
                ---
                """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    rec_audio = gr.Audio(
                        sources=["microphone", "upload"], 
                        type="filepath", 
                        label="🔴 บันทึกเสียง หรือ อัปโหลดไฟล์"
                    )
                with gr.Column(scale=1):
                    rec_api_key = gr.Textbox(
                        label="🔑 Typhoon API Key", 
                        type="password",
                        placeholder="sk-..."
                    )
            
            with gr.Row():
                rec_min_sec = gr.Slider(8, 15, value=11, step=1, label="Min Segment (วินาที)")
                rec_max_sec = gr.Slider(12, 20, value=15, step=1, label="Max Segment (วินาที)")
            
            rec_process_btn = gr.Button("🚀 สร้าง Dataset", variant="primary", size="lg")
            
            with gr.Row():
                rec_status = gr.Textbox(label="สถานะ", lines=3)
                rec_preview = gr.Textbox(label="Preview Transcriptions", lines=5)
            
            with gr.Row():
                rec_metadata = gr.Textbox(label="metadata.csv", lines=5)
                rec_zip = gr.File(label="📦 Download Dataset ZIP")
            
            rec_process_btn.click(
                process_recording_pipeline,
                inputs=[rec_audio, rec_api_key, rec_min_sec, rec_max_sec],
                outputs=[rec_status, rec_preview, rec_metadata, rec_zip]
            )

        # =====================
        # TAB 2: TRAIN
        # =====================
        with gr.Tab("2. 🏋️ Train Model"):
            gr.Markdown("""
            ### Fine-tune LoRA Model
            หลังจากสร้าง Dataset แล้ว กดปุ่ม Train เพื่อ fine-tune model
            loss ประมาณ 0.5 - 0.05 กำลังดี
            """)
            
            train_meta_path = gr.Textbox(
                label="📁 Path to metadata.csv", 
                value="dataset/metadata.csv"
            )
            
            with gr.Row():
                train_steps = gr.Number(label="Max Steps", value=600)
                train_lr = gr.Number(label="Learning Rate", value=2e-4)
            
            train_btn = gr.Button("🚀 Start Training", variant="primary")
            train_status = gr.Textbox(label="Training Status", lines=3)
            
            train_btn.click(
                trigger_training, 
                inputs=[train_meta_path, train_steps, train_lr], 
                outputs=train_status
            )
            
            gr.Markdown("""
            > ⚠️ **หมายเหตุ:** Training อาจใช้เวลา 10-30 นาที ขึ้นอยู่กับจำนวน steps และ GPU
            > ดู progress ได้ใน terminal/console
            """)

        # =====================
        # TAB 3: INFERENCE
        # =====================
        with gr.Tab("3. 🔊 Inference"):
            gr.Markdown("""
            ### สร้างเสียงจาก Text
            ใส่ข้อความที่ต้องการให้โมเดลพูด
            """)
            
            inf_model_path = gr.Textbox(
                label="📁 Model Path", 
                value="outputs/lora_model"
            )
            inf_text = gr.Textbox(
                label="📝 ข้อความที่ต้องการพูด", 
                lines=3,
                placeholder="สวัสดีครับ นี่คือเสียงที่สร้างจาก AI..."
            )
            
            inf_btn = gr.Button("🎵 Generate Audio", variant="primary")
            
            with gr.Row():
                inf_audio = gr.Audio(label="🔊 Generated Audio")
                inf_status = gr.Textbox(label="Status")
            
            inf_btn.click(
                run_inference, 
                inputs=[inf_text, inf_model_path], 
                outputs=[inf_audio, inf_status]
            )


if __name__ == "__main__":
    demo.launch(share=True)
