"""
Audio Processor Module
ตัดเสียงเป็น segments 11-15 วินาที และลบส่วนที่เงียบ
"""
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def remove_silence(audio_path, min_silence_len=500, silence_thresh=-40, keep_silence=300):
    """
    ลบส่วนที่เงียบจาก audio file
    
    Args:
        audio_path: path to audio file
        min_silence_len: minimum length of silence to detect (ms)
        silence_thresh: silence threshold in dB
        keep_silence: amount of silence to keep at start/end (ms)
    
    Returns:
        AudioSegment with silence removed
    """
    audio = AudioSegment.from_file(audio_path)
    
    # Detect non-silent parts
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    
    if not nonsilent_ranges:
        return audio
    
    # Combine non-silent parts with small padding
    combined = AudioSegment.empty()
    for start, end in nonsilent_ranges:
        # Add padding
        start = max(0, start - keep_silence)
        end = min(len(audio), end + keep_silence)
        combined += audio[start:end]
    
    return combined


def cut_audio_into_segments(audio_segment, min_sec=11, max_sec=15, target_sec=12):
    """
    ตัด audio เป็น segments ขนาด 11-15 วินาที
    
    Args:
        audio_segment: AudioSegment object
        min_sec: minimum segment length in seconds
        max_sec: maximum segment length in seconds
        target_sec: target segment length in seconds
    
    Returns:
        list of AudioSegment
    """
    duration_ms = len(audio_segment)
    target_ms = target_sec * 1000
    min_ms = min_sec * 1000
    max_ms = max_sec * 1000
    
    segments = []
    current_pos = 0
    
    while current_pos < duration_ms:
        remaining = duration_ms - current_pos
        
        if remaining <= max_ms:
            # Last segment - take all remaining if it's >= min_sec, else skip
            if remaining >= min_ms:
                segments.append(audio_segment[current_pos:])
            break
        
        # Try to find a good cut point (at silence)
        chunk = audio_segment[current_pos:current_pos + max_ms]
        
        # Find silence in the target range for better cuts
        silent_ranges = detect_nonsilent(
            chunk[target_ms:max_ms],
            min_silence_len=200,
            silence_thresh=-35
        )
        
        if silent_ranges and len(silent_ranges) > 0:
            # Cut at the start of the first non-silent part after target
            cut_point = target_ms + silent_ranges[0][0]
        else:
            # No good cut point found, use target length
            cut_point = target_ms
        
        segments.append(audio_segment[current_pos:current_pos + cut_point])
        current_pos += cut_point
    
    return segments


def process_audio_for_dataset(audio_path, output_dir, min_sec=11, max_sec=15):
    """
    ประมวลผล audio file สำหรับสร้าง dataset
    1. ลบส่วนที่เงียบ
    2. ตัดเป็น segments 11-15 วินาที
    3. บันทึกลง wavs folder
    
    Args:
        audio_path: path to input audio file
        output_dir: directory to save output files
        min_sec: minimum segment length
        max_sec: maximum segment length
    
    Returns:
        list of output file paths
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Remove silence
    print(f"[Audio] Removing silence from {audio_path.name}...")
    audio_no_silence = remove_silence(str(audio_path))
    
    # Step 2: Cut into segments
    print(f"[Audio] Cutting into {min_sec}-{max_sec}s segments...")
    segments = cut_audio_into_segments(audio_no_silence, min_sec, max_sec)
    
    print(f"[Audio] Created {len(segments)} segments")
    
    # Step 3: Save segments
    output_paths = []
    basename = audio_path.stem
    
    for i, segment in enumerate(segments, 1):
        output_name = f"{basename}_{i:03d}.wav"
        output_path = wavs_dir / output_name
        
        # Export as WAV 24kHz mono (optimal for TTS)
        segment = segment.set_frame_rate(24000).set_channels(1)
        segment.export(str(output_path), format="wav")
        
        output_paths.append(output_path)
        print(f"[Audio] Saved: {output_name} ({len(segment)/1000:.1f}s)")
    
    return output_paths


def get_audio_duration(audio_path):
    """Get audio duration in seconds"""
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        test_audio = sys.argv[1]
        output = process_audio_for_dataset(test_audio, "./test_output")
        print(f"Output files: {output}")
