import os
import math
import tempfile
import zipfile
import json
import shutil
from pathlib import Path
from datetime import datetime

import gradio as gr
import soundfile as sf
from transformers import pipeline

import requests
try:
    from speechbrain.inference.speaker import SpeakerRecognition
except (ImportError, AttributeError, RuntimeError, Exception) as e:
    print(f"SpeechBrain import failed: {e}")
    SpeakerRecognition = None

# ===============================
# 1) ASR MODEL
# ===============================
ASR_MODEL_NAME = "openai/whisper-small"

print(f"[ASR] Loading model: {ASR_MODEL_NAME}")
asr_pipe = pipeline(
    task="automatic-speech-recognition",
    model=ASR_MODEL_NAME,
    device_map="auto",
    return_timestamps=True,
)


# ===============================
# 2) HISTORY SYSTEM
# ===============================
HISTORY_DIR = Path(__file__).parent / "output_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_JSON = HISTORY_DIR / "history.json"


def load_history():
    if HISTORY_JSON.exists():
        try:
            return json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
        except:
            return []
    return []


def save_history(data):
    HISTORY_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_to_history(original_filename, zip_path, full_text, num_chunks):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    zip_path = Path(zip_path)
    new_zip = HISTORY_DIR / f"{timestamp_id}_{zip_path.name}"
    shutil.copy(zip_path, new_zip)

    record = {
        "id": timestamp_id,
        "timestamp": timestamp,
        "original_filename": original_filename,
        "zip_path": str(new_zip),
        "num_chunks": num_chunks,
        "preview": full_text[:200]
    }

    history = load_history()
    history.insert(0, record)
    save_history(history[:50])


def get_history_list():
    history = load_history()
    return [(f"[{h['timestamp']}] {h['original_filename']} ({h['num_chunks']} ท่อน)", h["id"]) for h in history]


def load_history_file(history_id):
    history = load_history()
    for h in history:
        if h["id"] == history_id:
            return h["zip_path"], f"✅ โหลด: {h['original_filename']}\n\n{h['preview']}"
    return None, "❌ ไม่พบไฟล์"


def delete_history_item(history_id):
    history = load_history()
    new = []

    for h in history:
        if h["id"] == history_id:
            try:
                Path(h["zip_path"]).unlink()
            except:
                pass
        else:
            new.append(h)

    save_history(new)
    return gr.update(choices=get_history_list(), value=None), "✅ ลบแล้ว"


# ===============================
# 3) FIX & CLEAN TIMESTAMP
# ===============================
def sanitize_timestamps(chunks, audio_duration):
    fixed = []
    last_end = 0.0

    for c in chunks:
        start, end = c["timestamp"]
        
        # Handle None values (Whisper can return None for end timestamp)
        if start is None:
            start = last_end
        if end is None:
            end = audio_duration

        if end <= start:
            end = start + 0.2

        start = max(0.0, min(start, audio_duration - 0.01))
        end = max(start + 0.01, min(end, audio_duration))

        if start < last_end:
            start = last_end
        if end <= start:
            end = start + 0.2

        fixed.append({"text": c["text"], "timestamp": (start, end)})
        last_end = end

    return fixed


# ===============================
# 4) MERGE CHUNKS
# ===============================
def merge_chunks_by_length(chunks, max_chars=80):
    merged = []
    cur_text = ""
    cur_start = None
    cur_end = None

    for c in chunks:
        text = c["text"].strip()
        start, end = c["timestamp"]

        if not text:
            continue

        if cur_start is None:
            cur_start = start

        if len(cur_text) + len(text) <= max_chars:
            cur_text += " " + text
            cur_end = end
        else:
            merged.append({"text": cur_text.strip(), "timestamp": (cur_start, cur_end)})
            cur_text = text
            cur_start = start
            cur_end = end

    if cur_text:
        merged.append({"text": cur_text.strip(), "timestamp": (cur_start, cur_end)})

    return merged


# ===============================
# 5) CUT AUDIO
# ===============================
def cut_audio_by_timestamps(audio_path, chunks, temp_dir, prefix):
    audio_path = Path(audio_path)
    data, sr = sf.read(audio_path)

    total_samples = len(data)
    outputs = []

    out_dir = temp_dir / "wavs"
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_duration = total_samples / sr  # Get audio duration in seconds

    for i, c in enumerate(chunks, 1):
        start, end = c["timestamp"]
        
        # Handle None timestamps
        if start is None:
            continue  # Skip chunks with no start time
        if end is None:
            end = audio_duration

        s = int(start * sr)
        e = int(end * sr)

        s = max(0, min(s, total_samples - 1))
        e = max(s + 1, min(e, total_samples))

        seg = data[s:e]

        out_name = f"{prefix}_{i:03d}.wav"
        out_path = out_dir / out_name
        sf.write(out_path, seg, sr)

        # Re-transcribe each segment individually to get accurate text
        segment_result = asr_pipe(
            {"array": seg, "sampling_rate": sr},
            return_timestamps=False,
            generate_kwargs={"task": "transcribe", "language": "th"}
        )
        segment_text = segment_result["text"].strip()
        
        # Use re-transcribed text, fallback to original if empty
        final_text = segment_text if segment_text else c["text"]
        outputs.append((out_path, final_text))

    return outputs, out_dir


# ===============================
# 6) TRANSCRIBE
# ===============================
def transcribe_with_timestamps(audio_path):
    data, sr = sf.read(audio_path)

    result = asr_pipe(
        {"array": data, "sampling_rate": sr},
        return_timestamps=True,
        generate_kwargs={"task": "transcribe", "language": "th"}
    )

    full_text = result["text"].strip()
    chunks = []

    for c in result.get("chunks", []):
        if c.get("timestamp") and c["text"].strip():
            chunks.append({
                "text": c["text"].strip(),
                "timestamp": c["timestamp"]
            })

    duration = len(data) / sr
    chunks = sanitize_timestamps(chunks, duration)

    return full_text, chunks


# ===============================
# 7) SPEAKER VERIFICATION
# ===============================
def verify_speaker(chunks, ref_audio_path, main_audio_path, temp_dir):
    if not ref_audio_path or SpeakerRecognition is None:
        return chunks

    print(f"[Speaker] Loading verification model...")
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")
    
    data, sr = sf.read(main_audio_path)
    filtered_chunks = []
    
    print(f"[Speaker] Verifying {len(chunks)} chunks...")
    for c in chunks:
        start, end = c["timestamp"]
        s, e = int(start * sr), int(end * sr)
        seg = data[s:e]
        
        # Save temp chunk
        tmp_chunk_path = temp_dir / "temp_chunk.wav"
        sf.write(tmp_chunk_path, seg, sr)
        
        score, prediction = verification.verify_files(ref_audio_path, str(tmp_chunk_path))
        if prediction: # True if match
            filtered_chunks.append(c)
            
    return filtered_chunks

# ===============================
# 8) TYPHOON API CORRECTION
# ===============================
def correct_text_typhoon(text, api_key):
    if not api_key: 
        return text
    
    url = "https://api.opentyphoon.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "typhoon-v1.5-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that corrects Thai text for spelling and grammar. Return ONLY the corrected text."},
            {"role": "user", "content": f"Correct this text: {text}"}
        ],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        res_json = response.json()
        if "choices" in res_json:
            return res_json["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Typhoon API Error: {e}")
        
    return text

# ===============================
# 9) MAIN PROCESS
# ===============================
def process_audio(audio_path, max_chars, ref_audio_path=None, typhoon_key=None):
    if not audio_path:
        return "ยังไม่ได้เลือกไฟล์", "", "", None

    audio_path = Path(audio_path)
    basename = audio_path.stem
    temp_dir = Path(tempfile.mkdtemp(prefix="tts_ds_"))

    full_text, chunks = transcribe_with_timestamps(audio_path)
    
    # Filter by Speaker
    if ref_audio_path:
        chunks = verify_speaker(chunks, ref_audio_path, audio_path, temp_dir)

    merged = merge_chunks_by_length(chunks, int(max_chars))
    
    # Correct Text with Typhoon
    if typhoon_key:
        print("[Typhoon] Correcting text...")
        for m in merged:
            m["text"] = correct_text_typhoon(m["text"], typhoon_key)

    pairs, segments_dir = cut_audio_by_timestamps(audio_path, merged, temp_dir, basename)

    metadata_lines = ["audio,text,source"]
    for p, txt in pairs:
        # Escape quotes and commas in text
        safe_txt = txt.replace('"', '""')
        metadata_lines.append(f"wavs/{p.name},\"{safe_txt}\",speaker")

    metadata_path = temp_dir / "metadata.csv"
    metadata_path.write_text("\n".join(metadata_lines), encoding="utf-8")

    zip_path = temp_dir / f"{basename}_tts_dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(metadata_path, "metadata.csv")
        for p in segments_dir.glob("*.wav"):
            zf.write(p, f"wavs/{p.name}")

    save_to_history(audio_path.name, zip_path, full_text, len(merged))

    preview = "\n".join([f"[{i+1}] {m['text']}" for i, m in enumerate(merged[:10])])

    return full_text, preview, "\n".join(metadata_lines), str(zip_path)


# ===============================
# 10) GRADIO UI
# ===============================
with gr.Blocks(title="Thai TTS Dataset Builder") as demo:
    gr.Markdown("# 🗣️ Thai TTS Dataset Builder")

    with gr.Tabs():
        with gr.Tab("🎤 สร้าง Dataset"):
            audio_in = gr.Audio(type="filepath", label="Main Audio File")
            ref_audio_in = gr.Audio(type="filepath", label="Reference Audio for Speaker Filtering (Optional)")
            typhoon_key = gr.Textbox(label="Typhoon API Key (Optional)", type="password")
            max_chars = gr.Slider(30, 200, 80, label="Max Characters per Chunk")

            run_btn = gr.Button("🚀 สร้าง")

            full_text_box = gr.Textbox(lines=6, label="Full Transcription")
            chunks_box = gr.Textbox(lines=8, label="Chunks")
            metadata_box = gr.Textbox(lines=10, label="Metadata (CSV)")
            zip_file = gr.File(label="Download Dataset")

            run_btn.click(
                fn=process_audio,
                inputs=[audio_in, max_chars, ref_audio_in, typhoon_key],
                outputs=[full_text_box, chunks_box, metadata_box, zip_file],
            )

        with gr.Tab("📜 ประวัติ"):
            history_dropdown = gr.Dropdown(choices=get_history_list())
            load_btn = gr.Button("📥 โหลด")
            delete_btn = gr.Button("🗑️ ลบ")
            history_status = gr.Textbox()
            history_file = gr.File()

            load_btn.click(
                fn=load_history_file,
                inputs=[history_dropdown],
                outputs=[history_file, history_status],
            )

            delete_btn.click(
                fn=delete_history_item,
                inputs=[history_dropdown],
                outputs=[history_dropdown, history_status],
            )


if __name__ == "__main__":
    demo.launch()
