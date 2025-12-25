"""
Typhoon Speech-to-Text Module
ใช้ Typhoon API (OpenAI-compatible) สำหรับ transcribe audio เป็นภาษาไทย
"""
import os
import requests
from pathlib import Path
from typing import List, Tuple, Optional


TYPHOON_ASR_ENDPOINT = "https://api.opentyphoon.ai/v1/audio/transcriptions"
TYPHOON_MODEL = "typhoon-v1-th"


def transcribe_audio(audio_path: str, api_key: str, model: str = TYPHOON_MODEL) -> str:
    """
    Transcribe single audio file using Typhoon ASR API
    
    Args:
        audio_path: path to audio file (WAV, MP3, etc.)
        api_key: Typhoon API key
        model: model name (default: typhoon-v1-th)
    
    Returns:
        transcribed text
    """
    if not api_key:
        raise ValueError("Typhoon API key is required")
    
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    with open(audio_path, "rb") as audio_file:
        files = {
            "file": (audio_path.name, audio_file, "audio/wav")
        }
        data = {
            "model": model,
            "language": "th"
        }
        
        try:
            response = requests.post(
                TYPHOON_ASR_ENDPOINT,
                headers=headers,
                files=files,
                data=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("text", "").strip()
            
        except requests.exceptions.RequestException as e:
            print(f"[Typhoon] API Error for {audio_path.name}: {e}")
            return ""


def transcribe_all_segments(
    wavs_dir: str, 
    api_key: str,
    progress_callback=None
) -> List[Tuple[str, str]]:
    """
    Transcribe all audio files in a directory
    
    Args:
        wavs_dir: directory containing WAV files
        api_key: Typhoon API key
        progress_callback: optional callback(current, total, filename)
    
    Returns:
        list of (filename, text) tuples
    """
    wavs_dir = Path(wavs_dir)
    wav_files = sorted(wavs_dir.glob("*.wav"))
    
    if not wav_files:
        print(f"[Typhoon] No WAV files found in {wavs_dir}")
        return []
    
    results = []
    total = len(wav_files)
    
    for i, wav_file in enumerate(wav_files, 1):
        print(f"[Typhoon] Transcribing ({i}/{total}): {wav_file.name}")
        
        if progress_callback:
            progress_callback(i, total, wav_file.name)
        
        text = transcribe_audio(str(wav_file), api_key)
        
        if text:
            results.append((wav_file.name, text))
            print(f"[Typhoon] Result: {text[:50]}...")
        else:
            print(f"[Typhoon] Warning: Empty result for {wav_file.name}")
    
    return results


def save_metadata_csv(
    transcriptions: List[Tuple[str, str]], 
    output_path: str,
    format_type: str = "ljspeech"
) -> str:
    """
    Save transcriptions to metadata.csv
    
    Args:
        transcriptions: list of (filename, text) tuples
        output_path: path to save metadata.csv
        format_type: "ljspeech" (path|text) or "csv" (audio,text,source)
    
    Returns:
        path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    
    if format_type == "ljspeech":
        # LJSpeech format: path|text
        for filename, text in transcriptions:
            # Clean text for LJSpeech format
            clean_text = text.replace("|", " ").replace("\n", " ").strip()
            lines.append(f"wavs/{filename}|{clean_text}")
    else:
        # CSV format with header
        lines.append("audio,text,source")
        for filename, text in transcriptions:
            # Escape quotes for CSV
            safe_text = text.replace('"', '""').replace("\n", " ").strip()
            lines.append(f'wavs/{filename},"{safe_text}",speaker')
    
    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    
    print(f"[Typhoon] Saved metadata to {output_path} ({len(transcriptions)} entries)")
    return str(output_path)


def process_wavs_to_metadata(
    wavs_dir: str,
    output_dir: str,
    api_key: str,
    format_type: str = "ljspeech",
    progress_callback=None
) -> str:
    """
    Full pipeline: transcribe all WAVs and save metadata.csv
    
    Args:
        wavs_dir: directory containing WAV files
        output_dir: directory to save metadata.csv
        api_key: Typhoon API key
        format_type: "ljspeech" or "csv"
        progress_callback: optional callback(current, total, filename)
    
    Returns:
        path to metadata.csv
    """
    # Transcribe all audio files
    transcriptions = transcribe_all_segments(wavs_dir, api_key, progress_callback)
    
    if not transcriptions:
        raise ValueError("No transcriptions generated")
    
    # Save metadata
    metadata_path = Path(output_dir) / "metadata.csv"
    save_metadata_csv(transcriptions, str(metadata_path), format_type)
    
    return str(metadata_path)


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 2:
        wavs_directory = sys.argv[1]
        api_key = sys.argv[2]
        result = process_wavs_to_metadata(wavs_directory, "./", api_key)
        print(f"Generated: {result}")
    else:
        print("Usage: python typhoon_stt.py <wavs_dir> <api_key>")
