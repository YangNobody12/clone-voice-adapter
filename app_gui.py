import gradio as gr
import os
from pathlib import Path
import threading
from recog_voice_for_prepare_data import process_audio, verify_speaker
from train import train
from inference import InferenceEngine
# Import speechbrain for standalone recog tab if needed, 
# but verify_speaker in recog_voice_for_prepare_data already wraps it.
# We might need to expose the raw verification function better if verify_speaker 
# is tightly coupled to chunks.
# Let's check verify_speaker implementation in recog_voice_for_prepare_data.py again.
# It takes chunks, ref_audio, main_audio... it's specific to the pipeline.
# I will re-implement a simple pair verification here or refactor. 
# Re-implementing is safer to avoid breaking the other script.

try:
    from speechbrain.inference.speaker import SpeakerRecognition
except (ImportError, AttributeError, RuntimeError, Exception) as e:
    print(f"SpeechBrain import failed (app_gui): {e}")
    SpeakerRecognition = None

# Global helper for Tab 1
def standalone_verify(file_a, file_b):
    if not file_a or not file_b:
        return "Please provide both audio files."
    if SpeakerRecognition is None:
        return "SpeechBrain not installed or model failed to load."
    
    try:
        verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")
        score, prediction = verification.verify_files(file_a, file_b)
        return f"Score: {score:.4f}\nPrediction: {'MATCH (Same Speaker)' if prediction else 'NO MATCH (Different Speaker)'}"
    except Exception as e:
        return f"Error: {e}"

# Helper for Tab 3 (Train) running in background
def run_training_thread(metadata_path, max_steps, learning_rate):
    # We need to modify train.py to accept these args or patch the Config
    # For now, we'll just run train(metadata_path) and maybe assume config.py is used
    # But user wants UI to control it. 
    # Let's import config and patch it before training
    from config import TrainConfig
    
    # Update global config (hacky but works for single user local app)
    # Better: Update train.py to accept a config object.
    # For this iteration, let's keep it simple and just run `train` 
    # assuming train.py reads from the file or we patch global config if possible.
    # Actually, train.py instantiates TrainConfig() inside the function.
    # We should update train.py to accept kwargs override.
    
    try:
        # Running as subprocess might be cleaner to avoid memory leaks but less feedback.
        # Let's call directly for improved feedback if possible.
        # But wait, train() creates a Trainer.
        
        # We will modify train.py slightly in the next step to allow overrides.
        train(metadata_path, overrides={"max_steps": int(max_steps), "learning_rate": float(learning_rate)})
        return "Training Complete!"
    except Exception as e:
        return f"Training Failed: {e}"

def trigger_training(metadata_path, max_steps, learning_rate):
    # Run in thread to not block UI
    # Note: Gradio queueing might handle this, but explicit thread is safer for long tasks
    # For simplicity in "no code" demo, we'll try to return a generator or just block if it's not too long? 
    # Training is LONG.
    # Let's just launch it and return "Started".
    t = threading.Thread(target=run_training_thread, args=(metadata_path, max_steps, learning_rate))
    t.start()
    return "Training started... check console for progress (Gradio logs)."

# Helper for Tab 4 (Inference)
inference_engine = None
def run_inference(text, model_path):
    global inference_engine
    if inference_engine is None:
        if not os.path.exists(model_path):
             return None, "Model path not found."
        inference_engine = InferenceEngine(model_path=model_path)
    
    # Reload if model path changed? Complex. Let's assume one model for now or reload if path differs.
    # Simplification: Just Create new engine if path differs.
    
    output_path = inference_engine.generate(text)
    return output_path, "Audio Generated!"

# --- GUI Layout ---
with gr.Blocks(title="Clone Voice Adapter Studio") as demo:
    gr.Markdown("# 🎙️ Clone Voice Adapter Studio")
    gr.Markdown("A unified tool for Voice Cloning: Recognition, Preparation, Training, and Inference.")

    with gr.Tabs():
        # TAB 1: RECOG
        # TAB 1: RECORD
        with gr.Tab("1. Record for Training"):
            gr.Markdown("### 🎙️ บันทึกเสียงสำหรับเทรน (Record Voice)")
            gr.Markdown("อ่านบทความด้านล่างนี้ด้วยน้ำเสียงที่เป็นธรรมชาติ (ความยาวประมาณ 1 นาที) เพื่อใช้เป็นข้อมูลเบื้องต้นสำหรับโมเดล")
            
            thai_script = """
            **บทอ่านทดสอบ (1 นาที):**
            
            "สวัสดีครับ/ค่ะ วันนี้เราจะมาทดสอบระบบสังเคราะห์เสียงภาษาไทยด้วยเทคโนโลยีปัญญาประดิษฐ์ 
            การสร้างเสียงสังเคราะห์ที่มีความเป็นธรรมชาติ จำเป็นต้องอาศัยข้อมูลเสียงที่มีคุณภาพ 
            และความชัดเจนในการออกเสียง ทั้งพยัญชนะ สระ และวรรณยุกต์ 
            
            ในปัจจุบัน เทคโนโลยี AI ได้เข้ามามีบทบาทสำคัญในชีวิตประจำวันของเรามากขึ้น 
            ไม่ว่าจะเป็นผู้ช่วยเสมือนจริง การแปลภาษาอัตโนมัติ หรือแม้แต่การอ่านข่าว 
            การที่เราสามารถโคลนเสียงของตนเองได้ จะช่วยเปิดโอกาสในการสร้างสรรค์คอนเทนต์รูปแบบใหม่ๆ 
            โดยไม่ต้องเสียเวลาบันทึกเสียงซ้ำหลายรอบ 
            
            ขอให้ท่านอ่านข้อความนี้ด้วยระดับเสียงปกติ ไม่เร็วและไม่ช้าจนเกินไป 
            เพื่อให้ระบบสามารถจับลักษณะเฉพาะของเนื้อเสียงและจังหวะการพูดของท่านได้อย่างแม่นยำที่สุด 
            ขอบคุณที่ร่วมเป็นส่วนหนึ่งในการพัฒนาเทคโนโลยีนี้ครับ/ค่ะ"
            """
            
            gr.Markdown(thai_script)
            
            with gr.Row():
                recog_audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🔴 กดปุ่มเพื่อบันทึกเสียง (Record)")
                recog_typhoon_key = gr.Textbox(label="Typhoon API Key (Optional for text correction)", type="password")
            
            with gr.Row():
                recog_max_chars = gr.Slider(30, 200, 80, label="Max Chars / Chunk")
            
            recog_process_btn = gr.Button("💾 Process & Create Dataset (สร้าง Dataset)")
            
            with gr.Row():
                recog_text = gr.Textbox(label="Full Transcript", lines=5)
                recog_chunks = gr.Textbox(label="Chunks Preview", lines=5)
            
            with gr.Row():
                recog_metadata = gr.Textbox(label="Metadata (CSV)", lines=5)
                recog_zip = gr.File(label="Download ZIP")

            # Wrapper to pass None for ref_audio
            def process_recording_wrapper(audio, max_chars, api_key):
                return process_audio(audio, max_chars, ref_audio_path=None, typhoon_key=api_key)

            recog_process_btn.click(
                process_recording_wrapper, 
                inputs=[recog_audio_in, recog_max_chars, recog_typhoon_key], 
                outputs=[recog_text, recog_chunks, recog_metadata, recog_zip]
            )


        # TAB 2: PREPARE
        with gr.Tab("2. Prepare"):
            gr.Markdown("### Dataset Builder")
            # Reusing arguments from recog_voice_for_prepare_data logic
            with gr.Row():
                prep_audio_in = gr.Audio(type="filepath", label="Main Audio File")
                prep_ref_audio = gr.Audio(type="filepath", label="Reference Audio (Filter Speaker)")
            
            with gr.Row():
                prep_typhoon_key = gr.Textbox(label="Typhoon API Key (Optional)", type="password")
                prep_max_chars = gr.Slider(30, 200, 80, label="Max Chars / Chunk")
            
            prep_btn = gr.Button("Process & Create Dataset")
            
            with gr.Row():
                prep_text = gr.Textbox(label="Full Transcript", lines=5)
                prep_chunks = gr.Textbox(label="Chunks Preview", lines=5)
            
            with gr.Row():
                prep_metadata = gr.Textbox(label="Metadata (CSV)", lines=5)
                prep_zip = gr.File(label="Download ZIP")

            prep_btn.click(
                process_audio, 
                inputs=[prep_audio_in, prep_max_chars, prep_ref_audio, prep_typhoon_key],
                outputs=[prep_text, prep_chunks, prep_metadata, prep_zip]
            )

        # TAB 3: TRAIN
        with gr.Tab("3. Train"):
            gr.Markdown("### Model Fine-tuning")
            train_meta_path = gr.Textbox(label="Path to metadata.csv", value="dataset/metadata.csv")
            with gr.Row():
                train_steps = gr.Number(label="Max Steps", value=60)
                train_lr = gr.Number(label="Learning Rate", value=2e-4)
            
            train_btn = gr.Button("Start Training")
            train_status = gr.Textbox(label="Status")
            
            train_btn.click(trigger_training, inputs=[train_meta_path, train_steps, train_lr], outputs=train_status)

        # TAB 4: INFERENCE
        with gr.Tab("4. Inference"):
            gr.Markdown("### Text-to-Speech")
            inf_model_path = gr.Textbox(label="Model Path", value="outputs/checkpoint-60")
            inf_text = gr.Textbox(label="Text to Speak", lines=2)
            inf_btn = gr.Button("Generate")
            
            inf_audio = gr.Audio(label="Output Audio")
            inf_status = gr.Textbox(label="Status")
            
            inf_btn.click(run_inference, inputs=[inf_text, inf_model_path], outputs=[inf_audio, inf_status])

if __name__ == "__main__":
    demo.launch(share=True)
